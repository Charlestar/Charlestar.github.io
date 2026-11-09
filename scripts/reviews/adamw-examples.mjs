import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const post = readFileSync('_posts/2026-11-10-adamw-optimizer-state-weight-decay.md', 'utf8');
function block(marker) {
  const expression = new RegExp(`<!-- example: ${marker} -->\\s*\x60\x60\x60python\\r?\\n([\\s\\S]*?)\\r?\\n\x60\x60\x60`, 'g');
  const matches = [...post.matchAll(expression)];
  assert.equal(matches.length, 1, `one ${marker} block required`);
  return matches[0][1];
}
const code = ['adamw-scalar', 'adamw-state-example'].map(block).join('\n');
const tests = String.raw`
import json
import random

def close(a, b):
    assert len(a) == len(b)
    assert all(math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-11)
               for x, y in zip(a, b)), (a, b)

# Oracle builds each moment from the complete participating gradient history.
# Missing gradients never enter that history, regardless of outer call count.
def history_oracle(theta, gradients, rates, beta1, beta2, decay, eps):
    history = []
    m = v = 0.0
    for grad, rate in zip(gradients, rates):
        if grad is None:
            continue
        history.append(grad)
        count = len(history)
        m = sum((1-beta1)*beta1**(count-i-1)*g
                for i, g in enumerate(history))
        v = sum((1-beta2)*beta2**(count-i-1)*g*g
                for i, g in enumerate(history))
        direction = (m/(1-beta1**count)) / (math.sqrt(v/(1-beta2**count))+eps)
        theta = theta - rate*decay*theta - rate*direction
    return theta, m, v, len(history)

rng = random.Random(1027)
for beta1 in (0.0, 0.5, 0.9):
    for beta2 in (0.0, 0.7, 0.99):
        for length in range(1, 15):
            gradients = [rng.choice((None, 0.0, rng.uniform(-3, 3))) for _ in range(length)]
            rates = [rng.choice((0.0, 0.01, 0.2)) for _ in gradients]
            state = (1.7, 0.0, 0.0, 0)
            for grad, rate in zip(gradients, rates):
                state = adamw_scalar(state, grad, lr=rate, beta1=beta1,
                                     beta2=beta2, weight_decay=0.03, eps=1e-8)
            close(state, history_oracle(1.7, gradients, rates, beta1, beta2, 0.03, 1e-8))

one = adamw_scalar((1.0, 0.0, 0.0, 0), 1.0)
two = adamw_scalar((2.0, 0.0, 0.0, 0), 1.0)
close((one[0], two[0]), (0.89, 1.88))
# Coupled L2 enters moments; its first normalized positive-gradient step differs.
coupled1 = adamw_scalar((1.0, 0.0, 0.0, 0), 1.1, weight_decay=0)
coupled2 = adamw_scalar((2.0, 0.0, 0.0, 0), 1.2, weight_decay=0)
close((coupled1[0], coupled2[0]), (0.9, 1.9))
assert not math.isclose(one[0], coupled1[0])
assert one[1:3] != coupled1[1:3]

assert adamw_scalar(one, None) is one
no_lr = adamw_scalar(one, -2.0, lr=0)
assert no_lr[0] == one[0] and no_lr[3] == one[3]+1
assert no_lr[1:3] != one[1:3]
zero_grad = adamw_scalar(one, 0.0)
assert zero_grad[0] < one[0] and zero_grad[3] == 2
assert zero_grad != adamw_scalar(one, None)

# Parameters share outer calls, but not participation counts or group settings.
per_parameter = [(1.0, 0.0, 0.0, 0), (1.0, 0.0, 0.0, 0)]
streams = [[1.0, None, 0.0, -1.0], [None, None, 2.0, None]]
groups = [dict(lr=0.1, weight_decay=0.1), dict(lr=0.03, weight_decay=0.0)]
for index in range(4):
    for parameter in range(2):
        per_parameter[parameter] = adamw_scalar(
            per_parameter[parameter], streams[parameter][index], **groups[parameter])
assert [state[3] for state in per_parameter] == [3, 1]
for parameter in range(2):
    settings = groups[parameter]
    close(per_parameter[parameter], history_oracle(
        1.0, streams[parameter], [settings['lr']]*4, 0.9, 0.99,
        settings['weight_decay'], 1e-12))
wrong_global_t = per_parameter[1][:3] + (4,)
assert not math.isclose(adamw_scalar(wrong_global_t, 1.0)[0],
                        adamw_scalar(per_parameter[1], 1.0)[0])

# Resume transports all four values, including the per-parameter count.
gradients = [1.0, None, 0.0, -2.0, 3.0, None, -0.1]
full = (0.7, 0.0, 0.0, 0)
for grad in gradients:
    full = adamw_scalar(full, grad)
for split in range(len(gradients)+1):
    state = (0.7, 0.0, 0.0, 0)
    for grad in gradients[:split]:
        state = adamw_scalar(state, grad)
    restored = tuple(json.loads(json.dumps(state)))
    for grad in gradients[split:]:
        restored = adamw_scalar(restored, grad)
    close(restored, full)
reset = adamw_scalar((one[0], 0.0, 0.0, 0), 0.0)
assert not math.isclose(reset[0], zero_grad[0])

# Constant gradient verifies initialization correction at every step.
constant = (1.0, 0.0, 0.0, 0)
for step in range(1, 21):
    constant = adamw_scalar(constant, 2.0, lr=0, weight_decay=0)
    close((constant[1]/(1-0.9**step), constant[2]/(1-0.99**step)), (2.0, 4.0))

# Pure decay product only applies when explicit gradients are supplied.
pure = (2.0, 0.0, 0.0, 0)
rates = [0.1, 0.05, 0, 0.2]
for rate in rates:
    pure = adamw_scalar(pure, 0.0, lr=rate, weight_decay=0.3)
assert math.isclose(pure[0], 2*math.prod(1-rate*0.3 for rate in rates))
correct = (1-0.1*0.1)*1 - 0.1*1
wrong = (1-0.1*0.1)*(1-0.1*1)
close((correct, wrong, wrong-correct), (0.89, 0.891, 0.001))

# Epsilon placement and algebraic rewriting are independent contracts.
small = adamw_scalar((0.0, 0.0, 0.0, 0), 1e-8,
                     lr=0.1, beta1=0, beta2=0, weight_decay=0, eps=1e-8)
assert math.isclose(small[0], -0.05)
inside = -0.1*1e-8/math.sqrt(1e-16+1e-8)
assert not math.isclose(small[0], inside)
for t in (1, 3, 10):
    m, v, eps = 0.012, 0.003, 0.07
    c1, c2 = 1-0.9**t, 1-0.99**t
    direct = (m/c1)/(math.sqrt(v/c2)+eps)
    rearranged = math.sqrt(c2)/c1*m/(math.sqrt(v)+eps*math.sqrt(c2))
    wrong_epsilon = math.sqrt(c2)/c1*m/(math.sqrt(v)+eps)
    assert math.isclose(direct, rearranged)
    assert not math.isclose(direct, wrong_epsilon)

# Independent exact-epsilon-zero history oracle: decoupled decay does not
# itself break invariance under one positive scale for the entire history.
gradient_path = [1.0, -0.4, 2.0, 0.0, -1.0]
rate_path = [0.1, 0.02, 0.05, 0.0, 0.03]
for prefix in range(1, len(gradient_path)+1):
    plain = history_oracle(1.0, gradient_path[:prefix], rate_path[:prefix],
                           0.9, 0.99, 0.2, 0.0)
    scaled = history_oracle(1.0, [7*g for g in gradient_path[:prefix]],
                            rate_path[:prefix], 0.9, 0.99, 0.2, 0.0)
    close((plain[0], plain[1]*7, plain[2]*49, plain[3]), scaled)

# The paper coefficient and PyTorch coefficient need a base-lr mapping.
base_lr, schedule, paper_decay, theta, direction = 0.001, 0.7, 0.02, 2.0, 0.3
paper_update = theta - schedule*(base_lr*direction + paper_decay*theta)
torch_update = (1-base_lr*schedule*(paper_decay/base_lr))*theta - base_lr*schedule*direction
assert math.isclose(paper_update, torch_update)

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('ValueError expected')

for keyword, value in [('lr', -0.1), ('beta1', 1), ('beta2', -0.1),
                       ('weight_decay', -0.1), ('eps', 0), ('eps', -1)]:
    rejects(lambda keyword=keyword, value=value: adamw_scalar(initial, 1, **{keyword:value}))
for grad in (float('nan'), float('inf'), True, '1'):
    rejects(lambda grad=grad: adamw_scalar(initial, grad))
for bad in [(1, 0, -1, 0), (1, 0, 0, -1), (1, 0, 0, True), (1, 0, 0, 0.5)]:
    rejects(lambda bad=bad: adamw_scalar(bad, 1))
rejects(lambda: adamw_scalar(initial, 1e308))
# Finite stored moments can normalize to infinity while theta stays finite.
rejects(lambda: adamw_scalar((1.0, 0.0, 1e308, 0), 0.0, beta2=0.999))
rejects(lambda: adamw_scalar(initial, 1e155))
for state, grad in [(initial, 10**1000), ((10**1000, 0, 0, 0), 1),
                    ((1, 0, 0, 10**1000), 1)]:
    rejects(lambda state=state, grad=grad: adamw_scalar(state, grad))
rejects(lambda: adamw_scalar(initial, 1, lr=10**300, weight_decay=10**300))
# A finite sqrt(v_hat) is tiny compared with the largest finite epsilon;
# adding it rounds back to epsilon rather than overflowing.
large_eps = adamw_scalar((1.0, 0.0, 1e308, 10), 0.0,
                         beta2=0.5, eps=1.7976931348623157e308)
assert math.isclose(large_eps[0], 0.99)

parameters = 10**9
assert 4*4*parameters == 16_000_000_000
assert 2*4*parameters == 8_000_000_000
assert (4*4+3*4)*parameters == 28_000_000_000
print('AdamW article examples: history-sum oracle, decay/L2, None/zero/lr0, epsilon, resume and byte models passed; CPU scalar evidence only.')
`;
const result = spawnSync(process.env.BLOG_PYTHON || 'python', ['-B', '-c', code + '\n' + tests], { encoding: 'utf8' });
if (result.error) throw result.error;
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0);
