import pandas as pd
from control import df
from Main import get_expected_transitions

# --- Export der Transition Abfolge zum vergleich---
transition_code = []
for _, row in df.iterrows():
    pid = row['Participant_ID']
    test = row['Test']
    
    # Konvertiere test zu String und handle NaN-Werte
    test_str = str(test) if pd.notna(test) else ''
    expected_transitions = get_expected_transitions(test_str)
    
    for t in row['transitions']:
        position = t.get('position', 0)
        expected_code = expected_transitions[position] if position < len(expected_transitions) else ''
        actual_code = t.get('transition_code', '')
        
        # Berechne Check-Wert (Differenz zwischen tatsächlich und erwartet)
        if actual_code != '' and expected_code != '':
            check = actual_code - expected_code
        else:
            check = ''
        
        transition_code.append({
            'Participant_ID': pid,
            'Test': test_str,
            'transition_code': actual_code,
            'expected_transition_code': expected_code,
            'check': check,
        })

df_transitions = pd.DataFrame(transition_code)
df_transitions.to_csv('safe_check_order.csv', index=False, encoding='utf-8-sig')
