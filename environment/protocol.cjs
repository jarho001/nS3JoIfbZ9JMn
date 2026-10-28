'use strict';

const AUDIENCES = new Set(['p1', 'p2', 'spectator']);

class ProtocolProjectionError extends Error {
	constructor(code) {
		super(`Showdown protocol projection rejected a packet (${code}).`);
		this.name = 'ProtocolProjectionError';
		this.code = code;
	}
}

function fail(code) {
	throw new ProtocolProjectionError(code);
}

function addEvent(records, message) {
	if (!message || message.startsWith('|t:|')) return;
	records.push({ type: 'event', message });
}

function projectUpdate(body, audience, records) {
	const lines = body.split('\n');
	let index = 0;
	while (index < lines.length) {
		const line = lines[index];
		if (line === '') {
			index++;
			continue;
		}
		if (line.startsWith('|split')) {
			const marker = /^\|split\|(p1|p2)$/.exec(line);
			if (!marker) fail('invalid_split_recipient');
			if (index + 2 >= lines.length) fail('truncated_split');
			const splitPlayer = marker[1];
			const secretLine = lines[index + 1];
			const publicLine = lines[index + 2];
			if (secretLine.startsWith('|split') || publicLine.startsWith('|split')) fail('nested_split');
			const visibleLine = audience === splitPlayer ? secretLine : publicLine;
			addEvent(records, visibleLine);
			index += 3;
			continue;
		}
		addEvent(records, line);
		index++;
	}
}

function projectSideUpdate(body, audience, records) {
	const separator = body.indexOf('\n');
	if (separator < 0) fail('malformed_sideupdate');
	const recipient = body.slice(0, separator);
	if (recipient !== 'p1' && recipient !== 'p2') fail('unsupported_sideupdate_recipient');
	const message = body.slice(separator + 1);
	if (message.startsWith('|split')) fail('unsupported_sideupdate_split');
	if (recipient !== audience) return;
	if (!message || message.includes('\n')) fail('malformed_sideupdate');
	if (message.startsWith('|t:|')) return;
	if (message.startsWith('|request|')) {
		let request;
		try {
			request = JSON.parse(message.slice('|request|'.length));
		} catch {
			fail('invalid_request_json');
		}
		if (!request || typeof request !== 'object' || Array.isArray(request)) fail('invalid_request_json');
		records.push({ type: 'request', request });
		return;
	}
	if (message.startsWith('|error|') || message.startsWith('|')) {
		records.push({ type: 'privateEvent', message });
		return;
	}
	fail('unsupported_private_event');
}

/**
 * Project complete BattleStream object packets to exactly one recipient.
 * The output is a versioned view sequence with no placeholders for packets
 * that recipient could not see. Inputs must already be complete packets in
 * the form "channel\nbody"; transport chunk reconstruction belongs upstream.
 */
function projectPackets(packets, audience) {
	if (!AUDIENCES.has(audience)) fail('unsupported_audience');
	if (!Array.isArray(packets)) fail('expected_complete_packet_array');
	const records = [];
	for (const packet of packets) {
		if (typeof packet !== 'string') fail('malformed_packet');
		const separator = packet.indexOf('\n');
		if (separator < 1) fail('malformed_packet');
		const channel = packet.slice(0, separator);
		const body = packet.slice(separator + 1);
		switch (channel) {
		case 'update':
			projectUpdate(body, audience, records);
			break;
		case 'sideupdate':
			projectSideUpdate(body, audience, records);
			break;
		case 'end':
			// End packets can contain complete trusted battle metadata and are not player views.
			break;
		case 'error':
			// The harness handles errors; untrusted error text is not actor input.
			fail('battle_error');
		default:
			fail('unsupported_channel');
		}
	}
	return { schemaVersion: 1, audience, records };
}

module.exports = { ProtocolProjectionError, projectPackets };
