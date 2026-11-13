"""Evaluate the generated straight-line Forza VS for captured float4 inputs.

Diagnostic only: does not model GPU rounding, interpolation or pixel shading.
Only use on trusted, locally generated HLSL; expressions are evaluated in Python.
"""
import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class V:
    def __init__(self, value):
        self.a = np.asarray(value, dtype=np.float32)

    def __getattr__(self, mask):
        if not mask or any(ch not in 'xyzw' for ch in mask):
            raise AttributeError(mask)
        return V(self.a[..., ['xyzw'.index(ch) for ch in mask]])

    def __add__(self, other): return V(self.a + other.a)
    def __neg__(self): return V(-self.a)


def mul(a, b):
    return V(np.where((a.a == 0) | (b.a == 0), 0, a.a * b.a))


def lit(v):
    a = v.a
    diffuse = np.maximum(a[..., 0], 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        spec = np.where(a[..., 3] == 0, 1,
                        np.exp2(np.clip(a[..., 3], -127.99609375, 127.99609375)
                                * np.log2(np.maximum(a[..., 1], 0))))
    return V(np.stack([np.ones_like(diffuse), diffuse,
                       np.where(diffuse > 0, spec, 0), np.ones_like(diffuse)], -1))


def evaluate(prefix):
    layout = prefix.with_name(prefix.name + '-mesh.txt').read_text()
    attributes = re.findall(r'attribute (\d+) format (\d+) buffer (\d+) offset (\d+) decode (\d+)', layout)
    assert 'buffer 0 stride 64' in layout
    assert all(fmt == '2' and buf == '0' and decode == '0'
               for _, fmt, buf, _, decode in attributes), 'Unsupported input layout'
    raw = np.fromfile(str(prefix) + '-vb-0.bin', dtype='<f4').reshape(-1, 16)
    n = len(raw)
    constants = np.fromfile(str(prefix) + '-vertex-constants.bin', dtype='<f4').reshape(192, 4)
    env = {'i': SimpleNamespace(**{'v'+r: V(raw[:, int(off)//4:int(off)//4+4])
                                   for r, _, _, off, _ in attributes}),
           'c': [V(row) for row in constants],
           'mac_mul': mul, 'ilu_lit': lit,
           'dot': lambda a, b: V(np.sum(a.a*b.a, axis=-1, keepdims=True)),
           'max': lambda a, b: V(np.maximum(a.a, b.a)),
           'min': lambda a, b: V(np.minimum(a.a, b.a)),
           'ilu_rsq': lambda v: V(np.repeat(1 / np.sqrt(np.abs(v.a)), 4, axis=-1)),
           'ilu_rcc': lambda v: V(np.repeat(np.copysign(np.clip(np.abs(1/v.a),
                                  5.42101e-20, 1.84467e19), 1/v.a), 4, axis=-1))}
    for name in [f'R{i}' for i in range(12)] + ['oPos', 'oD0', 'oD1', 'oB0', 'oB1', 'oFog', 'oPts'] + [f'oT{i}' for i in range(4)]:
        env[name] = V(np.zeros((n, 4)))
    for name in ['oD0', 'oB0', 'oT0', 'oT1', 'oT2', 'oT3']:
        env[name].a[:, 3] = 1
    source = Path(str(prefix) + '-vs.hlsl').read_text()
    body = source.split('    {', 1)[1].split('    VOut o;', 1)[0]
    for line in body.splitlines():
        expr = re.fullmatch(r'\s*precise float4 (mr|ir) = (.*);', line)
        if expr:
            env[expr[1]] = eval(expr[2], {'__builtins__': {}}, env)
            continue
        assignment = re.fullmatch(r'\s*(R\d+|o\w+)\.([xyzw]+) = (mr|ir)\.([xyzw]+);', line)
        if assignment:
            dest, mask, src, swizzle = assignment.groups()
            value = getattr(env[src], swizzle).a.copy()
            env[dest].a[:, ['xyzw'.index(ch) for ch in mask]] = value
        elif line.strip() not in ('', '{', '}'):
            raise ValueError('Unparsed instruction: ' + line)
    indices = np.fromfile(str(prefix) + '-indices.bin', dtype='<u4')
    used = np.unique(indices)
    result = {'draw': prefix.name, 'used_vertices': len(used)}
    arrays = {}
    for name in ['oPos', 'oD0', 'oD1', 'oT0', 'oT1', 'oT2', 'oT3']:
        val = env[name].a
        if name in ('oD0', 'oD1'): val = np.clip(val, 0, 1)
        arrays[name] = val
        result[name] = {'min': np.min(val[used], axis=0).tolist(),
                        'max': np.max(val[used], axis=0).tolist(),
                        'mean': np.mean(val[used], axis=0).tolist(),
                        'nonfinite': int(np.count_nonzero(~np.isfinite(val[used])))}
    np.savez(str(prefix)+'-evaluated.npz', **arrays, inputs=raw, indices=indices)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    reports = [evaluate(f.with_name(f.name.removesuffix('-mesh.txt')))
               for f in sorted(args.directory.glob('*-mesh.txt'))]
    out = args.directory / 'evaluated-summary.json'
    out.write_text(json.dumps(reports, indent=2))
    print(out)
    for r in reports:
        print(r['draw'], 'diffuse mean', r['oD0']['mean'], 'specular mean', r['oD1']['mean'])
