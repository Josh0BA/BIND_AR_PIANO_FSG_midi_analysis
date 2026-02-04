def read_vlq(data, pos):
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos

def get_state_sequence(filename, states):
    with open(filename, 'rb') as f:
        data = f.read()
    
    pos = 0
    if data[pos:pos+4] == b'MThd':
        pos += 4
        h_len = int.from_bytes(data[pos:pos+4], 'big')
        pos += 4 + h_len
    
    active_notes = set()
    sequence = []
    last_state = None
    all_midi_notes = set()
    
    while pos < len(data):
        if data[pos:pos+4] == b'MTrk':
            pos += 4
            t_len = int.from_bytes(data[pos:pos+4], 'big')
            pos += 4
            t_end = pos + t_len
            
            running_status = None
            while pos < t_end:
                delta, pos = read_vlq(data, pos)
                status = data[pos]
                if status & 0x80:
                    pos += 1
                    running_status = status
                else:
                    status = running_status
                
                msg_type = status & 0xF0
                if msg_type == 0x90: # Note On
                    note = data[pos]
                    vel = data[pos+1]
                    pos += 2
                    if vel > 0:
                        active_notes.add(note)
                        all_midi_notes.add(note)
                    else:
                        active_notes.discard(note)
                elif msg_type == 0x80: # Note Off
                    note = data[pos]
                    pos += 2
                    active_notes.discard(note)
                elif msg_type in [0xA0, 0xB0, 0xE0]:
                    pos += 2
                elif msg_type in [0xC0, 0xD0]:
                    pos += 1
                elif status == 0xFF:
                    pos += 1
                    m_len, pos = read_vlq(data, pos)
                    pos += m_len
                elif status in [0xF0, 0xF7]:
                    s_len, pos = read_vlq(data, pos)
                    pos += s_len
                
                # Identify current match
                current_match = None
                for s_id, s_notes in states.items():
                    if active_notes == s_notes:
                        current_match = s_id
                        break
                
                if current_match != last_state:
                    if current_match is not None:
                        sequence.append(current_match)
                    last_state = current_match
        else:
            pos += 1
    return sequence, all_midi_notes

STATE_DEFS = {
    0: {65, 67, 72, 74, 77, 79},
    1: {60, 62, 64, 72, 77, 79},
    2: {60, 62, 64, 67, 72, 76},
    3: {60, 62, 64, 76, 77, 79},
    4: {62, 64, 65, 76, 77, 79},
    5: {64, 65, 72, 76, 77, 79},
    6: {60, 62, 72, 74, 76, 77},
    7: {64, 65, 72, 74, 77, 79},
    8: {62, 64, 65, 67, 72, 74}
}

sequence_b1, notes_b1 = get_state_sequence('MIDI_BE13RE_B1.mid', STATE_DEFS)
from collections import Counter
counts_b1 = Counter(sequence_b1)

print(f"Notes in B1: {sorted(list(notes_b1))}")
print(f"Sequence B1: {sequence_b1}")
print(f"Counts B1: {counts_b1}")
