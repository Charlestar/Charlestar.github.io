#!/usr/bin/env python3
"""Strict CPU-only numerical evidence checks, not model/GPU/source certification.

Python standard library only. CLI: --self-test OR --input PATH.
v1 groups are independent; no tensor/UINT4/logits provenance is inferred.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


class EvidenceError(Exception):
    def __init__(self, code, stage, field, message, exit_status=1):
        super().__init__(message)
        self.code, self.stage, self.field = code, stage, field
        self.exit_status = exit_status

    def report(self):
        return {"status": "failed", "code": self.code, "stage": self.stage,
                "field": self.field, "message": str(self)}


def input_error(field, message):
    raise EvidenceError("input_error", "schema", field, message, 2)


def numeric_error(stage, field, message):
    raise EvidenceError("numeric_error", stage, field, message)


def checked(value, stage, field):
    if not math.isfinite(value):
        numeric_error(stage, field, "non-finite intermediate; outside binary64 contract")
    return value


def operation(function, stage, field):
    try:
        value = function()
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        numeric_error(stage, field, str(exc))
    return checked(value, stage, field)


def total(values, stage, field):
    return operation(lambda: math.fsum(values), stage, field)


def squared(value, stage, field):
    result = operation(lambda: value * value, stage, field)
    if value != 0 and result == 0:
        numeric_error(stage, field, "nonzero component square underflowed to zero")
    return result


def ratio(numerator, denominator, stage, field):
    result = operation(lambda: numerator / denominator, stage, field)
    if numerator != 0 and result == 0:
        numeric_error(stage, field, "nonzero ratio underflowed to zero")
    return result


def real(value, field):
    if type(value) not in (int, float):
        input_error(field, "expected a finite JSON number, not bool or a string")
    try:
        result = float(value)
    except OverflowError:
        input_error(field, "number outside binary64 input range")
    if not math.isfinite(result):
        input_error(field, "non-finite input")
    if type(value) is int and int(result) != value:
        input_error(field, "integer cannot be represented exactly in binary64")
    return result


def integer(value, field, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        input_error(field, "integer outside the declared range")
    return value


def fields(obj, required, allowed, field):
    if type(obj) is not dict:
        input_error(field, "expected an object")
    missing, extra = set(required) - obj.keys(), obj.keys() - set(allowed)
    if missing or extra:
        input_error(field, "missing=" + repr(sorted(missing)) + "; unknown=" + repr(sorted(extra)))


def vector(values, field):
    if type(values) is not list or not values:
        input_error(field, "expected a nonempty one-dimensional array")
    return [real(x, f"{field}[{i}]") for i, x in enumerate(values)]


def matrix(values, field):
    if type(values) is not list or not values:
        input_error(field, "expected a nonempty N by V matrix")
    rows = [vector(row, f"{field}[{i}]") for i, row in enumerate(values)]
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        input_error(field, "V must be >=2 and rows must not be ragged")
    return rows


def parse_document(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                input_error("$", "duplicate JSON key: " + repr(key))
            obj[key] = value
        return obj

    def constant(token):
        input_error("$", "nonstandard JSON constant: " + token)

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            input_error("$", "JSON number outside binary64 range")
        mantissa = token.lower().split("e", 1)[0]
        if value == 0 and any(digit in "123456789" for digit in mantissa):
            input_error("$", "nonzero JSON number underflowed to zero")
        return value

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                          parse_float=finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        input_error("$", "invalid UTF-8/JSON: " + str(exc))


def validate_document(doc):
    allowed = {"schema_version", "tensor", "logits", "uint4", "external_evidence"}
    fields(doc, {"schema_version"}, allowed, "$")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        input_error("schema_version", "only integer version 1 is supported")
    if not any(group in doc for group in ("tensor", "logits", "uint4")):
        input_error("$", "at least one computable data group is required")
    clean = {}
    if "external_evidence" in doc:
        if type(doc["external_evidence"]) is not dict:
            input_error("external_evidence", "expected an object")
        clean["external_evidence"] = doc["external_evidence"]
    if "tensor" in doc:
        obj = doc["tensor"]
        fields(obj, {"reference", "candidate"}, {"reference", "candidate"}, "tensor")
        ref, cand = vector(obj["reference"], "tensor.reference"), vector(
            obj["candidate"], "tensor.candidate")
        if len(ref) != len(cand):
            input_error("tensor", "reference and candidate lengths differ")
        clean["tensor"] = {"reference": ref, "candidate": cand}
    if "logits" in doc:
        obj = doc["logits"]
        keys = {"reference", "candidate", "targets", "mask"}
        fields(obj, keys, keys, "logits")
        ref, cand = matrix(obj["reference"], "logits.reference"), matrix(
            obj["candidate"], "logits.candidate")
        if len(ref) != len(cand) or len(ref[0]) != len(cand[0]):
            input_error("logits", "reference/candidate N by V shapes differ")
        n, width = len(ref), len(ref[0])
        if type(obj["targets"]) is not list or len(obj["targets"]) != n:
            input_error("logits.targets", "expected exactly N targets")
        targets = [integer(t, f"logits.targets[{i}]", 0, width - 1)
                   for i, t in enumerate(obj["targets"])]
        mask = obj["mask"]
        if type(mask) is not list or len(mask) != n or any(type(x) is not bool for x in mask):
            input_error("logits.mask", "expected exactly N JSON booleans")
        if not any(mask):
            input_error("logits.mask", "at least one scoring row is required")
        clean["logits"] = {"reference": ref, "candidate": cand, "targets": targets, "mask": mask}
    if "uint4" in doc:
        obj = doc["uint4"]
        keys = {"codes", "packed_bytes", "logical_length", "scale", "zero_point",
                "rounding", "layout"}
        fields(obj, keys, keys, "uint4")
        length = integer(obj["logical_length"], "uint4.logical_length", 1)
        if type(obj["codes"]) is not list or len(obj["codes"]) != length:
            input_error("uint4.codes", "codes length must equal logical_length")
        codes = [integer(c, f"uint4.codes[{i}]", 0, 15) for i, c in enumerate(obj["codes"])]
        if type(obj["packed_bytes"]) is not list or len(obj["packed_bytes"]) != (length + 1) // 2:
            input_error("uint4.packed_bytes", "byte count must be ceil(logical_length/2)")
        packed = [integer(b, f"uint4.packed_bytes[{i}]", 0, 255)
                  for i, b in enumerate(obj["packed_bytes"])]
        scale = real(obj["scale"], "uint4.scale")
        if scale <= 0:
            input_error("uint4.scale", "scale must be positive")
        zero = integer(obj["zero_point"], "uint4.zero_point", 0, 15)
        if obj["rounding"] != "ties_to_even" or obj["layout"] != "low_first":
            input_error("uint4", "only ties_to_even / low_first is supported")
        clean["uint4"] = {"codes": codes, "packed_bytes": packed, "logical_length": length,
                         "scale": scale, "zero_point": zero,
                         "rounding": "ties_to_even", "layout": "low_first"}
    return clean


def affine_quantize(values, scale, zero_point, qmin, qmax):
    # Internal scalar reference; the external v1 UINT4 group has no float source.
    low = operation(lambda: scale * (qmin - zero_point), "quantize", "low_endpoint")
    high = operation(lambda: scale * (qmax - zero_point), "quantize", "high_endpoint")
    codes, clamped, range_exceed = [], 0, 0
    for i, value in enumerate(values):
        quotient = operation(lambda: value / scale, "quantize", f"quotient[{i}]")
        raw_code = round(quotient) + zero_point
        clamped += not qmin <= raw_code <= qmax
        range_exceed += not low <= value <= high
        codes.append(min(qmax, max(qmin, raw_code)))
    n = len(values)
    return codes, {
        "code_clamp_fraction": clamped / n,
        "endpoint_fraction": sum(c in (qmin, qmax) for c in codes) / n,
        "prequant_range_exceed_fraction": range_exceed / n,
    }


def pack_uint4(codes):
    return [codes[i] | ((codes[i + 1] if i + 1 < len(codes) else 0) << 4)
            for i in range(0, len(codes), 2)]


def unpack_uint4(packed, logical_length):
    if logical_length % 2 and packed[-1] >> 4:
        raise EvidenceError("parity_error", "packing", "uint4.padding",
                            "unused high nibble must be zero")
    return [c for byte in packed for c in (byte & 15, byte >> 4)][:logical_length]


def uint4_metrics(obj):
    decoded = unpack_uint4(obj["packed_bytes"], obj["logical_length"])
    if decoded != obj["codes"]:
        raise EvidenceError("parity_error", "packing", "uint4.codes",
                            "decoded bytes do not match the supplied codes")
    rebuilt = [operation(lambda c=c: obj["scale"] * (c - obj["zero_point"]),
                         "dequantize", f"uint4[{i}]") for i, c in enumerate(decoded)]
    return {"status": "parity_pass", "contract": {
                "dtype": "UINT4", "rounding": "ties_to_even", "layout": "low_first",
                "scale": obj["scale"], "zero_point": obj["zero_point"],
                "logical_length": obj["logical_length"], "padding_high_nibble": 0},
            "byte_count": len(obj["packed_bytes"]), "dequantized": rebuilt,
            "endpoint_fraction": sum(c in (0, 15) for c in decoded) / len(decoded),
            "source_quantization_error": {"status": "skipped", "reason": "no original float source"},
            "source_authenticity": "unverified"}


def tensor_metrics(ref, cand):
    errors = [operation(lambda x=x, y=y: y - x, "tensor", f"error[{i}]")
              for i, (x, y) in enumerate(zip(ref, cand))]
    sse = total([squared(e, "tensor", "squared_error") for e in errors],
                "tensor", "sse")
    ref_sq = total([squared(x, "tensor", "reference_square") for x in ref],
                   "tensor", "reference_squared_norm")
    cand_sq = total([squared(x, "tensor", "candidate_square") for x in cand],
                    "tensor", "candidate_squared_norm")
    ref_zero, cand_zero = all(x == 0 for x in ref), all(x == 0 for x in cand)
    if (not ref_zero and ref_sq == 0) or (not cand_zero and cand_sq == 0):
        numeric_error("tensor", "squared_norm", "nonzero vector norm underflowed to zero")
    relative = None if ref_zero else ratio(sse, ref_sq, "tensor", "relative_squared_l2")
    cosine, cosine_reason = None, "zero norm" if ref_zero or cand_zero else None
    if cosine_reason is None:
        dot = total([operation(lambda x=x, y=y: x * y, "tensor", "dot_term")
                     for x, y in zip(ref, cand)], "tensor", "dot")
        denominator = operation(lambda: math.sqrt(ref_sq) * math.sqrt(cand_sq),
                                "tensor", "cosine_denominator")
        if denominator <= 0:
            numeric_error("tensor", "cosine_denominator", "positive denominator became zero")
        cosine = ratio(dot, denominator, "tensor", "cosine")
    return {"status": "computed", "count": len(ref),
            "mse": ratio(sse, len(ref), "tensor", "mse"),
            "max_absolute_error": max(abs(e) for e in errors),
            "relative_squared_l2": relative,
            "relative_squared_l2_reason": "zero reference norm" if ref_zero else None,
            "cosine": cosine, "cosine_reason": cosine_reason,
            "interpretation": "descriptive metrics, not an acceptance threshold"}


def log_softmax(row, field):
    peak = max(row)
    shifted = [operation(lambda x=x: x - peak, "log_softmax", field) for x in row]
    denominator = total([math.exp(x) for x in shifted], "log_softmax", field)
    if denominator <= 0:
        numeric_error("log_softmax", field, "normalization denominator is not positive")
    log_denominator = math.log(denominator)
    return [operation(lambda x=x: x - log_denominator, "log_softmax", field) for x in shifted]


def kl_from_logs(logp, logq, field):
    terms = []
    for lp, lq in zip(logp, logq):
        difference = operation(lambda: lp - lq, "kl", field)
        # For normalized probabilities, sum(q-p)=0. Evaluate the equivalent
        # generalized KL as nonnegative terms p*log(p/q)+q-p, rather than
        # cancelling positive/negative terms into a spurious negative KL.
        if abs(difference) < 1e-3:
            # d + expm1(-d), through degree 7; omitted terms are below
            # binary64 rounding at this threshold. Avoid cancellation at 0.
            kernel = difference * difference * (0.5 + difference * (
                -1/6 + difference * (1/24 + difference * (
                    -1/120 + difference * (1/720 - difference/5040)))))
            if difference != 0 and kernel == 0:
                numeric_error("kl", field, "nonzero KL kernel underflowed to zero")
            log_weight = lp
        elif difference >= 0:
            kernel = difference + math.expm1(-difference)
            log_weight = lp
        else:
            kernel = difference * math.exp(difference) - math.expm1(difference)
            log_weight = lq
        checked(kernel, "kl", field)
        if kernel < 0:
            numeric_error("kl", field, "nonnegative KL kernel became negative")
        # Do not first underflow exp(log_weight), then multiply by a large
        # kernel: the final weighted term can still be representable.
        terms.append(0.0 if kernel == 0 else operation(
            lambda: math.exp(log_weight + math.log(kernel)), "kl", field))
    return total(terms, "kl", field)


def nll_summary(losses, field):
    count = len(losses)
    summed = total(losses, "nll", field)
    mean = ratio(summed, count, "nll", field)
    ppl = operation(lambda: math.exp(mean), "perplexity", field)
    return {"nll_sum": summed, "scored_targets": count, "mean_nll": mean, "perplexity": ppl}


def logits_metrics(obj):
    kls, ref_losses, cand_losses, agreements = [], [], [], []
    for i, (ref, cand, target, score) in enumerate(zip(
            obj["reference"], obj["candidate"], obj["targets"], obj["mask"])):
        if not score:
            continue
        lp, lq = log_softmax(ref, f"reference[{i}]"), log_softmax(cand, f"candidate[{i}]")
        kls.append(kl_from_logs(lp, lq, f"row[{i}]"))
        ref_losses.append(-lp[target])
        cand_losses.append(-lq[target])
        # max returns the first index in a tie; both sides use the same convention.
        agreements.append(max(range(len(ref)), key=ref.__getitem__) ==
                          max(range(len(cand)), key=cand.__getitem__))
    n = len(kls)
    return {"status": "computed", "total_rows": len(obj["reference"]), "scored_rows": n,
            "vocabulary_size": len(obj["reference"][0]), "temperature": 1.0,
            "kl_reference_to_candidate_nats": ratio(total(kls, "kl", "sum"), n, "kl", "mean"),
            "top1_agreement": sum(agreements) / n, "top1_tie_break": "lowest index",
            "reference": nll_summary(ref_losses, "reference"),
            "candidate": nll_summary(cand_losses, "candidate"),
            "alignment": "caller-supplied next-token rows; no automatic shift",
            "semantic_alignment": "reported premise, not verified",
            "mask_scope": "same true rows for KL, top1, NLL and perplexity"}


def evaluate_document(doc, raw_sha256=None, mode="input"):
    clean = validate_document(doc)
    groups = {key: {"status": "skipped", "reason": "data group absent"}
              for key in ("tensor", "logits", "uint4")}
    if "tensor" in clean:
        groups["tensor"] = tensor_metrics(clean["tensor"]["reference"], clean["tensor"]["candidate"])
    if "logits" in clean:
        groups["logits"] = logits_metrics(clean["logits"])
    if "uint4" in clean:
        groups["uint4"] = uint4_metrics(clean["uint4"])
    return {"schema_version": 1, "tool": "quantization-acceptance", "mode": mode,
            "status": "incomplete", "cpu_checks_status": "completed",
            "input_sha256": raw_sha256, "results": groups,
            "verified_scope": [key + ": supplied CPU numeric data" for key in groups
                               if groups[key]["status"] != "skipped"],
            "unverified": ["source_authenticity", "cross_group_identity", "graph_rewrite",
                           "teacher_forcing_semantics", "real_gpu_kernel", "model_task_quality",
                           "service_performance"],
            "external_evidence": {"status": "reported_unverified",
                                  "metadata": clean.get("external_evidence", {})}}


def fixture():
    return {"schema_version": 1,
            "tensor": {"reference": [1, 2], "candidate": [2, 4]},
            "logits": {"reference": [[0, 0], [0, 0]],
                       "candidate": [[0, 0], [0, 2]],
                       "targets": [0, 0], "mask": [True, False]},
            "uint4": {"codes": [0, 1, 2], "packed_bytes": [16, 2], "logical_length": 3,
                      "scale": 0.5, "zero_point": 1, "rounding": "ties_to_even",
                      "layout": "low_first"},
            "external_evidence": {"kernel_pass": True, "model_revision": "not authenticated"}}


def self_test():
    passed = []

    def check(name, condition):
        if not condition:
            raise EvidenceError("selftest_error", "self_test", name, "assertion failed")
        passed.append(name)

    def expect(name, code, function):
        try:
            function()
        except EvidenceError as exc:
            if exc.code != code:
                raise EvidenceError("selftest_error", "self_test", name,
                                    "wrong failure code: " + exc.code) from exc
        else:
            raise EvidenceError("selftest_error", "self_test", name, "expected failure did not occur")
        passed.append(name)

    def mutation(group, key, value):
        doc = fixture()
        doc[group][key] = value
        return doc

    def fixture_reconstruction(doc):
        result = evaluate_document(doc)["results"]["uint4"]["dequantized"]
        if result != [-0.5, 0.0, 0.5]:
            raise EvidenceError("fixture_parity_error", "fixture", "dequantized",
                                "known fixture reconstruction differs")
        return result

    check("RNE half ties", [round(x) for x in (0.5, 1.5, 2.5, -0.5, -1.5)] ==
          [0, 2, 2, 0, -2])
    check("integer zero added after RNE", affine_quantize([0.5], 1, 1, 0, 15)[0] == [1]
          and round(0.5 + 1) == 2)
    known_bytes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
    check("fixed nibble oracle", pack_uint4(list(range(16))) == known_bytes)
    check("fixed unpack oracle", unpack_uint4(known_bytes, 16) == list(range(16)))
    check("odd logical length", pack_uint4([0, 1, 2]) == [0x10, 0x02] and
          unpack_uint4([0x10, 0x02], 3) == [0, 1, 2])
    report = evaluate_document(fixture(), mode="self_test")
    check("cosine does not imply amplitude", math.isclose(
        report["results"]["tensor"]["cosine"], 1.0) and
        report["results"]["tensor"]["mse"] == 2.5 and
        report["results"]["tensor"]["relative_squared_l2"] == 1.0)
    _, clamp = affine_quantize([0, 7, 7.2, 100], 1, 0, -7, 7)
    check("three distinct clamp fractions", clamp == {
        "code_clamp_fraction": 0.25, "endpoint_fraction": 0.75,
        "prequant_range_exceed_fraction": 0.5})
    check("one shared scoring mask", report["results"]["logits"]["scored_rows"] == 1 and
          report["results"]["logits"]["top1_agreement"] == 1.0 and
          report["results"]["logits"]["kl_reference_to_candidate_nats"] == 0.0 and
          math.isclose(report["results"]["logits"]["candidate"]["perplexity"], 2.0))
    one = {"schema_version": 1, "logits": {"reference": [[0, 0]], "candidate": [[0, 0]],
                                         "targets": [1], "mask": [True]}}
    check("N1 is one scored target", evaluate_document(one)["results"]["logits"]["scored_rows"] == 1)
    labels = [[-100, 2, 3, -100], [-100, -100, 3, 4]]
    check("count actual shifted labels", [sum(x != -100 for x in row[1:]) for row in labels] == [2, 2])
    weighted = nll_summary([math.log(2)] + [math.log(4)] * 3, "weighted")
    check("token weighted PPL", math.isclose(weighted["perplexity"], 3.363585661014858))
    lp, lq = [math.log(x) for x in (0.75, 0.25)], [math.log(0.5)] * 2
    check("KL direction", math.isclose(kl_from_logs(lp, lq, "test"), 0.13081203594113697)
          and math.isclose(kl_from_logs(lq, lp, "test"), 0.14384103622589045))
    near_ref = [2.452614615935712, -0.10910710576120053, 3.6311162686894303,
                2.9003006069414257]
    near_cand = [2.4526146219971254, -0.10910710290931795, 3.631116278287903,
                 2.9003006017458595]
    check("small positive KL without signed cancellation", math.isclose(
        kl_from_logs(log_softmax(near_ref, "near_ref"), log_softmax(near_cand, "near_cand"),
                     "near"), 1.976138120905224e-17, rel_tol=1e-6))
    check("weighted KL remains representable after tiny probability", math.isclose(
        kl_from_logs(log_softmax([0.0, -1000.0], "tiny_ref"),
                     log_softmax([0.0, -1e308], "tiny_cand"), "tiny_weighted"),
        5.075958897549457e-127, rel_tol=1e-12, abs_tol=0.0))
    extreme = {"schema_version": 1, "logits": {
        "reference": [[0, 0, -1000]], "candidate": [[0, -1000, 0]],
        "targets": [0], "mask": [True]}}
    er = evaluate_document(extreme)["results"]["logits"]
    check("stable KL despite exp underflow", math.isclose(er["kl_reference_to_candidate_nats"], 500)
          and math.isclose(er["reference"]["perplexity"], 2.0))
    zero = tensor_metrics([0.0], [0.0])
    check("zero norm undefined", zero["relative_squared_l2"] is None and zero["cosine"] is None)
    check("known reconstruction", fixture_reconstruction(fixture()) == [-0.5, 0, 0.5])
    check("external claims cannot verify GPU", report["status"] == "incomplete" and
          report["external_evidence"]["status"] == "reported_unverified" and
          "real_gpu_kernel" in report["unverified"])
    raw = b'{"schema_version":1,"tensor":{"reference":[1],"candidate":[1]}}\n'
    check("strict JSON parse", evaluate_document(parse_document(raw))["results"]["tensor"]["mse"] == 0)
    check("raw bytes identity is whitespace sensitive",
          hashlib.sha256(raw).hexdigest() != hashlib.sha256(raw.rstrip()).hexdigest())
    expect("swapped nibble", "parity_error", lambda: evaluate_document(
        mutation("uint4", "packed_bytes", [1, 2])))
    expect("nonzero padding", "parity_error", lambda: evaluate_document(
        mutation("uint4", "packed_bytes", [16, 18])))
    expect("wrong byte count", "input_error", lambda: evaluate_document(
        mutation("uint4", "packed_bytes", [16])))
    expect("wrong logical length", "input_error", lambda: evaluate_document(
        mutation("uint4", "logical_length", 4)))
    expect("bool code", "input_error", lambda: evaluate_document(
        mutation("uint4", "codes", [False, 1, 2])))
    expect("bool target including masked row", "input_error", lambda: evaluate_document(
        mutation("logits", "targets", [0, False])))
    expect("masked out target still checked", "input_error", lambda: evaluate_document(
        mutation("logits", "targets", [0, 2])))
    expect("masked out NaN still checked", "input_error", lambda: evaluate_document(
        mutation("logits", "candidate", [[0, 0], [float("nan"), 0]])))
    expect("non-boolean mask", "input_error", lambda: evaluate_document(
        mutation("logits", "mask", [1, 0])))
    expect("empty scoring mask", "input_error", lambda: evaluate_document(
        mutation("logits", "mask", [False, False])))
    expect("zero scale", "input_error", lambda: evaluate_document(mutation("uint4", "scale", 0)))
    expect("float zero point", "input_error", lambda: evaluate_document(
        mutation("uint4", "zero_point", 1.0)))
    expect("ragged matrix", "input_error", lambda: evaluate_document(
        mutation("logits", "candidate", [[0, 0], [0]])))
    expect("unknown group field", "input_error", lambda: evaluate_document(
        mutation("tensor", "typo", 1)))
    expect("bool version", "input_error", lambda: evaluate_document({"schema_version": True}))
    expect("inexact computational integer", "input_error",
           lambda: tensor_metrics([real(2**53 + 1, "integer")], [float(2**53)]))
    check("exact large computational integer", real(2**53 + 2, "integer") == 2**53 + 2)
    expect("metadata alone", "input_error", lambda: evaluate_document(
        {"schema_version": 1, "external_evidence": {"kernel_pass": True}}))
    expect("duplicate nested key", "input_error", lambda: parse_document(
        b'{"x":{"a":1,"a":2}}'))
    expect("nonstandard NaN", "input_error", lambda: parse_document(b'{"x":NaN}'))
    expect("float literal overflow", "input_error", lambda: parse_document(b'{"x":1e400}'))
    expect("nonzero float literal underflow", "input_error", lambda: parse_document(b'{"x":1e-400}'))
    expect("metadata float overflow", "input_error", lambda: parse_document(
        b'{"external_evidence":{"x":1e400}}'))
    expect("strict UTF8", "input_error", lambda: parse_document(
        '{"schema_version":1}'.encode("utf-16")))
    expect("division overflow", "numeric_error", lambda: affine_quantize(
        [1e308], 1e-308, 0, 0, 15))
    expect("difference overflow", "numeric_error", lambda: tensor_metrics([1e308], [-1e308]))
    expect("square overflow", "numeric_error", lambda: tensor_metrics([1e200], [0.0]))
    expect("norm underflow", "numeric_error", lambda: tensor_metrics([1e-300], [1e-300]))
    expect("mixed magnitude squared error underflow", "numeric_error",
           lambda: tensor_metrics([1.0, 1e-200], [1.0, 0.0]))
    tiny_error = float.fromhex("0x1.0000000000000p-537")
    expect("positive MSE underflow", "numeric_error",
           lambda: tensor_metrics([0.0, 0.0], [tiny_error, 0.0]))
    expect("positive relative error underflow", "numeric_error",
           lambda: tensor_metrics([2.0, 0.0], [2.0, tiny_error]))
    for sign in (1, -1):
        expect("signed cosine ratio underflow " + str(sign), "numeric_error",
               lambda sign=sign: tensor_metrics([1.0, 1.0, tiny_error],
                                                 [1.0, -1.0, sign * tiny_error]))
    huge_scale = mutation("uint4", "scale", 1e308)
    huge_scale["uint4"]["zero_point"] = 0
    expect("DQ overflow", "numeric_error", lambda: evaluate_document(huge_scale))
    expect("PPL overflow", "numeric_error", lambda: nll_summary([1000.0], "test"))
    expect("logits span overflow", "numeric_error", lambda: log_softmax([1e308, -1e308], "test"))
    expect("legal changed scale fails only known fixture", "fixture_parity_error",
           lambda: fixture_reconstruction(mutation("uint4", "scale", 1.0)))
    expect("legal changed zero fails only known fixture", "fixture_parity_error",
           lambda: fixture_reconstruction(mutation("uint4", "zero_point", 2)))
    changed = fixture()
    changed["uint4"]["codes"] = [1, 1, 2]
    changed["uint4"]["packed_bytes"] = [17, 2]
    check("consistent altered data is not authenticated",
          evaluate_document(changed)["results"]["uint4"]["status"] == "parity_pass")
    check("strict JSON report", json.loads(json.dumps(report, allow_nan=False))["status"] == "incomplete")
    report["self_test"] = {"status": "passed", "checks": len(passed), "names": passed,
                           "fixtures": "synthetic CPU data only"}
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="run synthetic CPU checks")
    group.add_argument("--input", metavar="PATH", help="read one strict UTF-8 v1 JSON file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            report = self_test()
        else:
            try:
                raw = Path(args.input).read_bytes()
            except (OSError, ValueError) as exc:
                raise EvidenceError("input_error", "read", "input", str(exc), 2) from exc
            report = evaluate_document(parse_document(raw), hashlib.sha256(raw).hexdigest())
        serialized = json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True)
    except EvidenceError as exc:
        print(json.dumps(exc.report(), ensure_ascii=True, allow_nan=False), file=sys.stderr)
        return exc.exit_status
    except (AssertionError, OverflowError, ValueError, RecursionError) as exc:
        # Fail closed if a serialization/runtime boundary escaped the explicit checks.
        print(json.dumps({"status": "failed", "code": "runtime_error",
                          "stage": "runtime", "message": str(exc)}, ensure_ascii=True,
                         allow_nan=False), file=sys.stderr)
        return 1
    print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
