// Recomputable arithmetic/state counterexamples, NOT GPU/RDMA measurements.
// These models do not implement NIC ordering, CUDA visibility or a provider API.
import assert from 'node:assert/strict';

let passed = 0;
const check = (name, fn) => { fn(); console.log(`PASS ${++passed}: ${name}`); };
function permutations(xs) {
  if (!xs.length) return [[]];
  return xs.flatMap((x, i) => permutations(xs.filter((_, j) => i !== j)).map(p => [x, ...p]));
}

check('write-with-immediate notification consumes conventional RC receive credits', () => {
  let credits = 2, completedNotifications = 0;
  for (let message = 0; message < 3; message++) {
    if (!credits) break; // RNR: this model makes no claim about partial payload placement.
    credits--;
    completedNotifications++;
  }
  assert.equal(completedNotifications, 2);
  assert.equal(credits, 0);
  credits++; // Posting a new receive restores one notification credit.
  credits--; completedNotifications++;
  assert.equal(completedNotifications, 3);
});

check('logical generation rejection does not prevent late DMA corruption', () => {
  const orders = permutations(['oldDMA', 'reuseForB', 'newDMA'])
    .filter(p => p.indexOf('reuseForB') < p.indexOf('newDMA'));
  let corrupt = 0, safeAfterDrain = 0;
  for (const order of orders) {
    let generation = 1, value = 'A';
    for (const event of order) {
      if (event === 'reuseForB') { generation = 2; value = 'B-empty'; }
      if (event === 'newDMA') value = 'B';
      if (event === 'oldDMA') value = 'A-late'; // Same valid arena rkey; no NIC epoch check.
    }
    assert.equal(generation === 1, false); // Old completion is rejected in every order.
    if (value !== 'B') corrupt++;
    if (order.indexOf('oldDMA') < order.indexOf('reuseForB')) {
      assert.equal(value, 'B'); safeAfterDrain++;
    }
  }
  assert.equal(orders.length, 3);
  assert.equal(corrupt, 1);
  assert.equal(safeAfterDrain, 1);
});

check('completed revocation plus draining differs from metadata unpublication', () => {
  const liveKeys = new Set(['old']);
  let outstanding = 1, metadataPublished = false;
  assert.equal(metadataPublished, false);
  assert.equal(liveKeys.has('old'), true); // Directory deletion did not change NIC authority.
  const canReuse = () => !liveKeys.has('old') && outstanding === 0;
  liveKeys.delete('old'); // Abstract *completed* provider revocation, not API submission.
  assert.equal(canReuse(), false); // Already accepted DMA still needs draining.
  outstanding--;
  assert.equal(canReuse(), true);
  liveKeys.add('new');
  assert.equal(liveKeys.has('old'), false);
});

check('two Simple slices can advance four FIFO steps, including an empty tail', () => {
  const chunkSteps = 4, sliceSteps = 2, chunkSize = 4096;
  const sliceSize = chunkSize / chunkSteps * sliceSteps;
  assert.equal(sliceSize, 2048);
  const starts = [], sizes = [];
  let step = 0, remaining = 1024;
  for (let slice = 0; slice < chunkSteps / sliceSteps; slice++) {
    starts.push(step);
    sizes.push(Math.min(remaining, sliceSize));
    remaining -= sizes.at(-1);
    step += sliceSteps;
  }
  assert.deepEqual(starts, [0, 2]);
  assert.deepEqual(sizes, [1024, 0]);
  assert.equal(step, 4);
});

check('nonblocking revoke must reach success; any non-busy value is not sufficient', () => {
  const mayUseQuiescedResources = state => state === 'ncclSuccess';
  assert.deepEqual(['ncclInProgress', 'ncclInProgress', 'ncclSuccess']
    .map(mayUseQuiescedResources), [false, false, true]);
  assert.equal(mayUseQuiescedResources('ncclSystemError'), false);
});

check('NCCL protocol payload ratios are encodings, not measured bus bandwidth', () => {
  const llBytes = [4, 4, 4, 4]; // data, flag, data, flag.
  assert.equal((llBytes[0] + llBytes[2]) / llBytes.reduce((a, b) => a + b), 0.5);
  const ll128Words = Array.from({ length: 16 }, (_, i) => i !== 15);
  assert.equal(ll128Words.filter(Boolean).length / ll128Words.length, 0.9375);
});

check('Mooncake v0.3.12 explicit runtime setting overrides true default', () => {
  const getBool = value => value === undefined ? true : ['1', 'true', 'on', 'yes'].includes(value.toLowerCase());
  assert.equal(getBool(undefined), true);
  assert.equal(getBool('0'), false);
  assert.equal(getBool('1'), true);
  assert.equal(getBool('ON'), true);
});

console.log(`${passed} network examples passed; no hardware performance or API conformance claim.`);
