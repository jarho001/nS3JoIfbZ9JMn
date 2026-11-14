"""Lossless host journal adapter for the shared pure JAX transition kernel.

CoreState is the device carry; a Checkpoint also owns every recipient history.
Pages freeze event-time rendering so later species changes cannot rewrite it.
"""
from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
import jax
import numpy as np
from . import shared_engine as e
from . import shared_engine_io as io
from . import gen5_tables as tables

CORE_SCHEMA=18

@dataclass(frozen=True)
class Page:
    rows: tuple
    recipients: tuple

@dataclass(frozen=True)
class Checkpoint:
    core: e.CoreState
    pages: tuple


def from_state(state):
    if not isinstance(state,e.State):raise ValueError('initial migration requires complete State history')
    e._validate_structure(state)
    rows=tuple(tuple(map(int,row))for row in np.asarray(state.history)[:int(state.hcount)])
    views=tuple(tuple(ev['message']for ev in io.player_view(state,side)['events'])for side in range(2))
    return Checkpoint(e.core_state(state),(Page(rows,views),))


def append(checkpoint,result):
    """Commit a successful core result and its events as one immutable value."""
    if not isinstance(result.state,e.CoreState):raise ValueError('core result required')
    code=int(result.code)
    if code not in (e.OK,e.PENDING,e.RETRY):return checkpoint
    rows=tuple(tuple(map(int,row))for row in np.asarray(result.trace.events)[:int(result.trace.ecount)])
    views=tuple(tuple(ev['message']for ev in io.project_events(result.state,result.trace,side))for side in range(2))
    pages=checkpoint.pages+((Page(rows,views),)if rows else ())
    return Checkpoint(result.state,pages)


def advance(checkpoint,choices,token=None,*,step=e.advance_core):
    result=step(checkpoint.core,choices,token)
    return append(checkpoint,result),result


def player_view(checkpoint,side):
    if side not in (0,1):raise ValueError('side must be 0 or 1')
    return {'request':io.project_request(checkpoint.core,side),
            'events':[{'type':'event','message':line}for page in checkpoint.pages for line in page.recipients[side]]}


def _checksum(payload,fields,names=None):
    digest=hashlib.sha256(payload.encode())
    for name in (e.CoreState._fields if names is None else names):
        value=np.asarray(fields[name]);digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def save(path,checkpoint):
    e._validate_structure(checkpoint.core)
    metadata={'journal_schema':CORE_SCHEMA,'data_sha256':tables.DATA_SHA256,
              'pages':[{'rows':page.rows,'recipients':page.recipients}for page in checkpoint.pages]}
    payload=json.dumps(metadata,sort_keys=True,separators=(',',':'))
    fields={k:np.asarray(v)for k,v in zip(e.CoreState._fields,checkpoint.core)}
    digest=_checksum(payload,fields)
    target=Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix=target.name+'.',delete=False)as file:
            temporary=Path(file.name)
            np.savez(file,journal_sha256=np.asarray(digest),metadata=np.asarray(payload),**fields)
            file.flush();os.fsync(file.fileno())
        os.replace(temporary,target)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def load(path,*,replayed=None):
    with np.load(path,allow_pickle=False)as data:
        payload=str(data['metadata'])
        version=json.loads(payload)['journal_schema']
        names=tuple(n for n in e.CoreState._fields if not(version==1 and n=='request_state') and not(version<6 and n=='item_provenance') and not(version<7 and n in e.identity_ops.FIELDS) and not(version<8 and n=='illusion') and not(version<10 and n=='ability_staleness') and not(version<13 and n in e.called_ops.FIELDS) and not(version<16 and n=='ability_seek'))
        if version not in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,CORE_SCHEMA):raise ValueError('unsupported core journal schema')
        if version<CORE_SCHEMA and replayed is None:raise ValueError('old core journal requires independent replay/provenance')
        if _checksum(payload,data,names)!=str(data['journal_sha256']):raise ValueError('journal checksum mismatch')
        metadata=json.loads(payload)
        if metadata['data_sha256']!=tables.DATA_SHA256:raise ValueError('journal schema/table mismatch')
        if set(data.files)!={'metadata','journal_sha256',*names}:raise ValueError('invalid core field set')
        if version<CORE_SCHEMA:
            e._validate_structure(replayed.core)
            for name in names:
                expected=np.asarray(getattr(replayed.core,name))
                if name=='volatile_state':expected=expected[:,:,:28 if version>=17 else 27 if version>=15 else 26 if version>=12 else 25 if version>=11 else 24 if version>=9 else 23 if version>=5 else 22 if version==4 else 21 if version==3 else 20,:]
                if name=='call_frames' and version<14:expected=expected[:,:16]
                if name=='action_state':expected=expected[:,:,:24 if version>=5 else 21]
                if data[name].shape!=expected.shape or data[name].dtype!=expected.dtype or not np.array_equal(data[name],expected):raise ValueError('core replay/provenance mismatch')
            core=replayed.core
        else:core=e.CoreState(*(e.jnp.asarray(data[name])for name in names))
        e._validate_structure(core)
        pages=[]
        for page in metadata['pages']:
            rows=tuple(tuple(row)for row in page['rows']);views=tuple(tuple(v)for v in page['recipients'])
            if any(len(row)!=(10 if version<8 else e.EVENT_COLUMNS) or any(type(v)is not int for v in row)for row in rows)or len(views)!=2 or any(type(v)is not str for view in views for v in view):raise ValueError('invalid journal page')
            pages.append(Page(rows,views))
        if not pages:raise ValueError('missing initialization journal')
        if version<CORE_SCHEMA:
            expected=tuple(Page(tuple(row[:10 if version<8 else e.EVENT_COLUMNS] for row in p.rows),p.recipients)for p in replayed.pages)
            if tuple(pages)!=expected:raise ValueError('journal replay/provenance mismatch')
            pages=list(replayed.pages)
        return Checkpoint(core,tuple(pages))
