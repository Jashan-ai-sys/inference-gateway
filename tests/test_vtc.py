from gateway.vtc import VTC


def test_pick_lowest_counter_with_name_tiebreak():
    vtc = VTC()
    assert vtc.pick([]) is None
    assert vtc.pick(["b", "a"]) == "a"
    vtc.charge_prompt("a", 10)
    assert vtc.pick(["a", "b"]) == "b"


def test_charges_are_weighted():
    vtc = VTC(prompt_weight=1.0, output_weight=2.0)
    vtc.charge_prompt("a", 100)
    vtc.charge_output("a", 10)
    assert vtc.counter("a") == 120


def test_returning_idle_tenant_is_lifted_to_lowest_backlogged_counter():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)  # a has been busy
    vtc.on_backlogged("b", backlogged=["a"])  # b returns from idle with counter 0
    assert vtc.counter("b") == 1000


def test_lift_never_lowers_a_counter():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)
    vtc.charge_prompt("b", 2000)
    vtc.on_backlogged("b", backlogged=["a"])
    assert vtc.counter("b") == 2000


def test_no_lift_when_nobody_else_is_backlogged():
    vtc = VTC()
    vtc.charge_prompt("a", 1000)
    vtc.on_backlogged("b", backlogged=[])
    assert vtc.counter("b") == 0


def test_returning_tenant_alternates_instead_of_monopolising():
    vtc = VTC()
    for _ in range(50):  # a served alone for a while
        vtc.charge_prompt("a", 100)
    vtc.on_backlogged("b", backlogged=["a"])
    served = []
    for _ in range(10):
        t = vtc.pick(["a", "b"])
        served.append(t)
        vtc.charge_prompt(t, 100)
    assert served.count("a") == 5 and served.count("b") == 5
