"""C14-PREFLIGHT-01: shared preflight for long runs on real hosts (AGENTS.md "Preflight before long runs"; roadmap C14).

An experiment runner calls run_preflight() before any real-host run expected to take more than 30 minutes, and starts
the run only if the result says passed. Over every host of every cohort the run may touch it checks:
  1. Host-level return gates, read from the named detector source: every `return make_res(<id>, "REJECT_..." or
     "SKIP_...", ...)` inside the analysis function (and, since C7, its listed stages) ends the whole host.
     gate_inventory() extracts each one with the conditions of the `if` statements that lead to it, and the
     function's literal parameter defaults. run_preflight()
     evaluates those exact condition texts on the host's metadata values with a restricted evaluator (comparisons,
     and/or/not, arithmetic, subscripts, and the calls len, abs, min, max, float, int, np.isfinite, math.isfinite).
     A gate whose values are missing is NOT_EVALUATED and fails, unless the run exempts it with a reason. A host that a
     gate rejects fails, unless its cohort allows gate rejects. A new gate in the source therefore fails every preflight
     until the run declares how it is checked.
  2. Inputs: every path exists. Non-blinded files are hashed against the expected SHA256; blinded files
     carry a declared digest only, without reading contents.
  3. Ranges and counts the run declares: each range must have lo < hi, each count must reach its minimum. A check may
     carry the preregistered reason that a named host is allowed to fail it; it is then recorded as allowed.
  4. Draws: every planned injection family is drawn within the bounded number of attempts.
Held-out and blinded cohorts are checked from metadata only: for them the tool never calls data_checks or draw. It
cannot see what host_values and metadata_checks read, so a runner must give them metadata only (limitation).
An exception in any callable is recorded with its type and message and fails the host; nothing is skipped silently.
It writes only its own result file, refuses to overwrite one, and writes nothing else.

Usage as a library: import preflight; res = preflight.run_preflight(...); preflight.passing_hosts(res, cohort).
Command line: python -I -B preflight.py inventory SOURCE [--function NAME] [--helper NAME]
"""
import ast, datetime, hashlib, json, math, os, sys, tempfile
from pathlib import Path

GATE_PREFIXES = ('REJECT_', 'SKIP_')
FUNCTION = '_analyze_single_target_impl'
HELPER = 'make_res'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class MissingValue(Exception):
    pass


class UnsupportedExpression(Exception):
    pass


def _isfinite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


CALLS = {'len': len, 'abs': abs, 'min': min, 'max': max, 'float': float, 'int': int,
         'np.isfinite': _isfinite, 'numpy.isfinite': _isfinite, 'math.isfinite': _isfinite}
COMPARE = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
           ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
           ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b}
BINOP = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
         ast.Div: lambda a, b: a / b}


def evaluate(expr, values):
    """Evaluate one gate condition text on a dict of values. Raises MissingValue for a name not in values and
    UnsupportedExpression for anything outside the restricted grammar; never calls eval."""
    def ev(n):
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                for v in n.values:
                    if not ev(v):
                        return False
                return True
            for v in n.values:
                if ev(v):
                    return True
            return False
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right in zip(n.ops, n.comparators):
                r = ev(right)
                if type(op) not in COMPARE or not COMPARE[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(n, ast.BinOp) and type(n.op) in BINOP:
            return BINOP[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in values:
                raise MissingValue(n.id)
            return values[n.id]
        if isinstance(n, ast.Subscript):
            base = ev(n.value)
            key = n.slice.value if isinstance(n.slice, ast.Constant) else None
            if not isinstance(n.slice, ast.Constant):
                raise UnsupportedExpression(ast.unparse(n))
            try:
                return base[key]
            except (KeyError, IndexError, TypeError):
                raise MissingValue(ast.unparse(n))
        if isinstance(n, ast.Call) and not n.keywords and ast.unparse(n.func) in CALLS:
            return CALLS[ast.unparse(n.func)](*[ev(a) for a in n.args])
        raise UnsupportedExpression(ast.unparse(n))
    return bool(ev(ast.parse(expr, mode='eval').body))


def names_in(expr):
    """Free variable names of a condition (call targets such as np.isfinite and len are not variables)."""
    tree = ast.parse(expr, mode='eval')
    funcs = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
    inner = {id(x) for c in ast.walk(tree) if isinstance(c, ast.Call) for x in ast.walk(c.func)}
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in funcs | inner})


def detector_stages(tree):
    """RC2-FREEZE-01 C7: the literal DETECTOR_STAGES of a detector source ({function or class: runs once per BLS
    peak}), or None for a source without it (the detector before C7 was one function)."""
    for n in tree.body:
        if isinstance(n, ast.Assign) and [getattr(t, 'id', None) for t in n.targets] == ['DETECTOR_STAGES']:
            return ast.literal_eval(n.value)
    return None


def gate_inventory(source, function=FUNCTION, helper=HELPER):
    """Every host-level return gate of `function` in `source`: status, line, the joined conditions of the enclosing
    `if` statements (negated when the return is in an else branch), the variable names, whether it sits inside a
    loop, and the literal parameter defaults. Also lists the other returns of the function for the reviewer.

    When the source lists `function` in DETECTOR_STAGES (C7), every listed stage is read, a gate counts as in the
    loop when its stage runs once per peak, and `helper` also matches its host form (host.make_res, self.make_res)
    and host.end_host, whose second argument is the status as well."""
    path = Path(source)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
    if len(fns) != 1:
        raise ValueError(f'{function} found {len(fns)} times in {path}')
    fn = fns[0]
    stages = detector_stages(tree)
    stages = stages if stages and function in stages else {function: False}
    helpers = {helper, 'host.' + helper, 'self.' + helper, 'host.end_host'} if len(stages) > 1 else {helper}
    scan = []  # (stage function or method, runs once per peak)
    for name, per_peak in stages.items():
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and n.name == name]
        if len(nodes) != 1:
            raise ValueError(f'detector stage {name} found {len(nodes)} times in {path}')
        scan += [(m, per_peak) for m in (nodes[0].body if isinstance(nodes[0], ast.ClassDef) else nodes)
                 if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    gates, other = [], []
    for stage, per_peak in scan:
        parent = {c: n for n in ast.walk(stage) for c in ast.iter_child_nodes(n)}

        def owner(node):
            p = node
            while p in parent:
                p = parent[p]
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return p
            return stage
        for n in ast.walk(stage):
            if not isinstance(n, ast.Return):
                continue
            call = n.value
            status = None
            if (isinstance(call, ast.Call) and ast.unparse(call.func) in helpers and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
                status = call.args[1].value
            nested = owner(n) is not stage  # a return inside a nested helper does not end the host
            if nested or not (status and status.startswith(GATE_PREFIXES)):
                other.append(dict(line=n.lineno, text=ast.unparse(n)[:100],
                                  nested_function=getattr(owner(n), 'name', '<lambda>') if nested else None))
                continue
            conds, loop, p = [], per_peak, n
            while p in parent:
                q = parent[p]
                if isinstance(q, ast.If):
                    conds.append(f'({ast.unparse(q.test)})' if p in q.body else f'(not ({ast.unparse(q.test)}))')
                if isinstance(q, (ast.For, ast.AsyncFor, ast.While)):
                    loop = True
                p = q
            cond = ' and '.join(reversed(conds)) or 'True'
            gates.append(dict(status=status, line=n.lineno, condition=cond, names=names_in(cond), in_loop=loop))
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = {}
    for arg, d in list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + [
            (k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]:
        if isinstance(d, ast.Constant) and isinstance(d.value, (int, float, str, bool)):
            defaults[arg.arg] = d.value
    return dict(source=str(path), sha256=sha256(path), function=function, function_line=fn.lineno, helper=helper,
                stages=sorted(stages),
                gates=sorted(gates, key=lambda g: g['line']), defaults=defaults,
                other_returns=sorted(other, key=lambda r: r['line']))


def _call(fn, *args):
    """Run one runner callable; an exception becomes a recorded error, never a silent skip."""
    try:
        return fn(*args), None
    except Exception as e:  # recorded in the host record and fails the preflight
        return None, f'{type(e).__name__}: {e}'


def _check_ranges(spec, rec, where):
    """A range is (lo, hi) and a count (value, minimum). A third element names the preregistered reason that this
    host may fail the check: it is then recorded as ALLOWED with that reason and does not fail the host."""
    for kind, items in (('range', (spec or {}).get('ranges', {})), ('count', (spec or {}).get('counts', {}))):
        for name, t in items.items():
            a, b = t[0], t[1]
            allowed = str(t[2]).strip() if len(t) > 2 and t[2] is not None else ''
            ok = a is not None and b is not None and (a < b if kind == 'range' else a >= b)
            entry = dict(lo=a, hi=b) if kind == 'range' else dict(value=a, minimum=b)
            entry.update(ok=ok, from_=where, allowed=allowed or None)
            rec['ranges' if kind == 'range' else 'counts'][name] = entry
            if not ok and not allowed:
                rec['failures'].append(f'range {name} empty or undefined ({a}, {b})' if kind == 'range'
                                       else f'count {name} = {a} below {b}')


def check_host(cohort, host, blinded, inv, *, host_values, exempt, allow_reject, inputs, metadata_checks,
               data_checks, draw, max_attempts, hash_cache):
    rec = dict(host=host, gates={}, inputs=[], ranges={}, counts={}, draws={}, failures=[],
               data_checks='NOT_RUN (blinded)' if blinded else None, draw='NOT_RUN (blinded)' if blinded else None)
    vals, err = _call(host_values, cohort, host)
    if err:
        rec['failures'].append(f'host_values raised {err}')
        vals = {}
    values = dict(inv['defaults'])
    values.update(vals or {})
    for g in inv['gates']:
        st = g['status']
        reason = exempt.get(f'{cohort}:{st}') or exempt.get(st)
        if reason:
            rec['gates'][st] = dict(result='EXEMPT', reason=reason)
            continue
        try:
            hit = evaluate(g['condition'], values)
            rec['gates'][st] = dict(result='REJECT' if hit else 'PASS')
            if hit and not allow_reject:
                rec['failures'].append(f'gate {st} (line {g["line"]}) rejects this host')
        except MissingValue as e:
            rec['gates'][st] = dict(result='NOT_EVALUATED', missing=str(e))
            rec['failures'].append(f'gate {st} not evaluated: missing {e}')
        except UnsupportedExpression as e:
            rec['gates'][st] = dict(result='NOT_EVALUATED', unsupported=str(e))
            rec['failures'].append(f'gate {st} not evaluated: unsupported expression {e}')
        except Exception as e:  # for example a comparison of a None value; recorded and failing
            rec['gates'][st] = dict(result='NOT_EVALUATED', error=f'{type(e).__name__}: {e}')
            rec['failures'].append(f'gate {st} not evaluated: {type(e).__name__}: {e}')
    if inputs:
        items, err = _call(inputs, cohort, host)
        if err:
            rec['failures'].append(f'inputs raised {err}')
        for it in items or []:
            p = Path(it['path'])
            entry = dict(path=str(it['path']), expected=it.get('sha256'), present=p.is_file())
            if blinded:
                entry['verification'] = 'NOT_RUN (blinded; manifest digest only)'
                digest = entry['expected']
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
                    rec['failures'].append(f'input SHA256 declaration missing or invalid: {it["path"]}')
            elif entry['present']:
                key = str(p.resolve())
                if key not in hash_cache:
                    hash_cache[key] = sha256(p)
                entry['sha256'] = hash_cache[key]
                entry['match'] = entry['expected'] is None or entry['sha256'] == entry['expected']
            if not entry['present']:
                rec['failures'].append(f'input missing: {it["path"]}')
            elif not blinded and not entry['match']:
                rec['failures'].append(f'input SHA256 mismatch: {it["path"]}')
            rec['inputs'].append(entry)
    if metadata_checks:
        spec, err = _call(metadata_checks, cohort, host)
        if err:
            rec['failures'].append(f'metadata_checks raised {err}')
        _check_ranges(spec, rec, 'metadata')
    if not blinded:
        if data_checks:
            spec, err = _call(data_checks, cohort, host)
            if err:
                rec['failures'].append(f'data_checks raised {err}')
            _check_ranges(spec, rec, 'data')
            rec['data_checks'] = 'RUN'
        if draw:
            fams, err = _call(draw, cohort, host)
            if err:
                rec['failures'].append(f'draw raised {err}')
            for fam, d in (fams or {}).items():
                ok = bool(d.get('drawn')) and (max_attempts is None or d.get('attempts', 0) <= max_attempts)
                rec['draws'][fam] = dict(d, ok=ok)
                if not ok:
                    rec['failures'].append(f'family {fam} not drawn within {max_attempts} attempts ({d})')
            rec['draw'] = 'RUN'
    rec['passed'] = not rec['failures']
    return rec


def run_preflight(*, source, cohorts, host_values, out, expected_source_sha256=None, exempt=None,
                  allow_gate_rejects=(), inputs=None, metadata_checks=None, data_checks=None, draw=None,
                  max_attempts=None, function=FUNCTION, helper=HELPER, note=None):
    """cohorts: {name: dict(hosts=[...], blinded=bool)}. host_values(cohort, host) -> dict of gate variable values
    (metadata only). exempt: {status or 'cohort:status': reason}. allow_gate_rejects: cohort names whose hosts may
    be rejected by a gate (reported, not failing). inputs(cohort, host) -> [dict(path, sha256 or None)].
    metadata_checks / data_checks(cohort, host) -> dict(ranges={name: (lo, hi)}, counts={name: (value, minimum)}).
    draw(cohort, host) -> {family: dict(drawn=bool, attempts=int, ...)}. data_checks and draw are never called for a
    blinded cohort. Blinded inputs have presence and manifest digest declarations checked, without content hashing. Writes the result JSON to `out` (refused if it exists) and returns it."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'preflight result {out} exists; each preflight writes a new file')
    exempt = dict(exempt or {})
    inv = gate_inventory(source, function, helper)
    failures = []
    if expected_source_sha256 and inv['sha256'] != expected_source_sha256:
        failures.append(f'source SHA256 {inv["sha256"][:16]} differs from expected {expected_source_sha256[:16]}')
    known = {g['status'] for g in inv['gates']}
    for key, reason in exempt.items():
        st = key.split(':', 1)[-1]
        if st not in known:
            failures.append(f'exemption for {key}: no such gate in {inv["function"]}')
        if not str(reason or '').strip():
            failures.append(f'exemption for {key} has no reason')
    hash_cache, res_cohorts = {}, {}
    for name, c in cohorts.items():
        blinded = bool(c.get('blinded'))
        recs = [check_host(name, h, blinded, inv, host_values=host_values, exempt=exempt,
                           allow_reject=name in allow_gate_rejects, inputs=inputs, metadata_checks=metadata_checks,
                           data_checks=data_checks, draw=draw, max_attempts=max_attempts, hash_cache=hash_cache)
                for h in c['hosts']]
        res_cohorts[name] = dict(blinded=blinded, n_hosts=len(recs), n_passed=sum(r['passed'] for r in recs),
                                 failed_hosts=[r['host'] for r in recs if not r['passed']],
                                 gate_rejects={g['status']: [r['host'] for r in recs
                                                             if r['gates'].get(g['status'], {}).get('result') == 'REJECT']
                                               for g in inv['gates']},
                                 hosts=recs)
    passed = not failures and all(not c['failed_hosts'] for c in res_cohorts.values())
    res = dict(tool='agent_exchange/preflight_01/preflight.py', tool_sha256=sha256(__file__),
               created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), note=note,
               passed=passed, failures=failures, inventory=inv, exempt=exempt,
               allow_gate_rejects=sorted(allow_gate_rejects), max_attempts=max_attempts, cohorts=res_cohorts)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + out.name + '.', dir=out.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, default=str, allow_nan=True)
        if out.exists():
            raise FileExistsError(f'preflight result {out} appeared during the preflight')
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return res


def passing_hosts(result, cohort):
    """Hosts of one cohort that passed; control hosts are chosen from these (AGENTS.md)."""
    return [r['host'] for r in result['cohorts'][cohort]['hosts'] if r['passed']]


def main(argv):
    if len(argv) >= 2 and argv[0] == 'inventory':
        fn = argv[argv.index('--function') + 1] if '--function' in argv else FUNCTION
        hp = argv[argv.index('--helper') + 1] if '--helper' in argv else HELPER
        print(json.dumps(gate_inventory(argv[1], fn, hp), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
