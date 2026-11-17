import json, glob
res = '/home/claude/axi-runtime/docs/results/'
M, SRC = {}, {}
for f in sorted(glob.glob(res + 'axi_model_ladder_complete_compact_stage*.json')):
    d = json.load(open(f))
    for tag, cells in d['models'].items():
        for c, v in cells.items():
            M.setdefault(tag, {})[c] = v['summary']; SRC[(tag, c)] = 'raw+summary'
    for c, v in d.get('screen_lines_only', {}).get('cells', {}).items():
        if c.endswith('_first_run'): continue
        tag = 'Qwen_Qwen2.5-32B-Instruct-AWQ'
        if c in M.get(tag, {}): continue
        M.setdefault(tag, {})[c] = {'exact_share': v['exact'] / 100, 'computed_per_step': v['computed'],
                                    'steps_per_hour': v['steps_h'], 'steps_done': v['steps'],
                                    'stopped': v.get('stopped'), 'cached_share': v.get('cached', 0) / 100}
        SRC[(tag, c)] = 'screen line only'
names = {'Qwen_Qwen2.5-1.5B-Instruct': '1.5B', 'Qwen_Qwen2.5-14B-Instruct-AWQ': '14B', 'Qwen_Qwen2.5-32B-Instruct-AWQ': '32B'}
ran = [t for t in names if t in M]
def g(t, c, k): return M[t][c][k]
def ex(t, c): return round(100 * g(t, c, 'exact_share'), 2)
for t in ran: print('==', names[t], {c: SRC[(t, c)] for c in sorted(M[t])})
print('1  cells present', {names[t]: len(M[t]) for t in ran})
print('2  PN', {names[t]: ex(t, 'PN') for t in ran})
print('3  P4', {names[t]: ex(t, 'P4') for t in ran})
def pcinfo(t):
    s = M[t]['PC']
    if 'steps_per_job' in s: return (min(s['steps_per_job']), max(s['steps_per_job']), s['stopped_jobs'] if 'stopped_jobs' in s else None)
    return ('lines', s['steps_done'], s['stopped'])
print('4  PC steps min/max per job, stopped', {names[t]: pcinfo(t) for t in ran})
print('5  PC/P4 computed per step', {names[t]: round(g(t, 'PC', 'computed_per_step') / g(t, 'P4', 'computed_per_step'), 3) for t in ran})
print('6  P4/PC steps/h', {names[t]: round(g(t, 'P4', 'steps_per_hour') / g(t, 'PC', 'steps_per_hour'), 3) for t in ran})
print('7  P0', {names[t]: ex(t, 'P0') for t in ran})
print('8  X0', {names[t]: ex(t, 'X0') for t in ran})
print('9  XW', {names[t]: ex(t, 'XW') for t in ran})
print('10 XR-XW', {names[t]: round(ex(t, 'XR') - ex(t, 'XW'), 2) for t in ran})
print('11 XR', {names[t]: ex(t, 'XR') for t in ran})
ref = {'P0': 94.9, 'X0': 84.0, 'XW': 47.8, 'XR': 90.1}
t = 'Qwen_Qwen2.5-14B-Instruct-AWQ'
print('12 14B vs Test 21', {c: round(ex(t, c) - r, 2) for c, r in ref.items()})
lad = [('1.5B', ex('Qwen_Qwen2.5-1.5B-Instruct', 'XW')), ('3B', 7.47), ('7B', 41.53), ('14B', ex('Qwen_Qwen2.5-14B-Instruct-AWQ', 'XW'))]
if 'Qwen_Qwen2.5-32B-Instruct-AWQ' in M: lad.append(('32B', ex('Qwen_Qwen2.5-32B-Instruct-AWQ', 'XW')))
print('13 XW ladder', lad, 'rises', all(lad[i][1] < lad[i + 1][1] for i in range(len(lad) - 1)))
print('against AXI: PN<95', {names[t]: ex(t, 'PN') < 95 for t in ran}, ' P4 computed >= PC', {names[t]: g(t, 'P4', 'computed_per_step') >= g(t, 'PC', 'computed_per_step') for t in ran})
