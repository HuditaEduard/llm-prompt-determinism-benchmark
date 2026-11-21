import json
import math
import random
from pathlib import Path

# Fixed seed guarantees 100% reproducibility across machines and runs
RANDOM_SEED = 42


def generate_digital_logic_tasks(n: int = 20) -> list[dict]:
    """Generates multi-stage Boolean logic gate evaluation problems."""
    tasks = []
    gates = ["AND", "OR", "NAND", "NOR", "XOR"]

    for i in range(1, n + 1):
        a, b, c = random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)
        g1, g2 = random.choice(gates), random.choice(gates)

        def eval_gate(gate: str, x: int, y: int) -> int:
            return {
                "AND": x & y,
                "OR": x | y,
                "NAND": int(not (x & y)),
                "NOR": int(not (x | y)),
                "XOR": x ^ y,
            }[gate]

        stage1 = eval_gate(g1, a, b)
        output = eval_gate(g2, stage1, c)

        tasks.append({
            "id": f"logic_{i:02d}",
            "domain": "digital_logic",
            "task": (
                f"A 2-stage digital logic circuit takes binary inputs A={a}, B={b}, and C={c}. "
                f"Inputs A and B feed into a 2-input {g1} gate to produce intermediate signal X. "
                f"Signals X and C then feed into a 2-input {g2} gate to produce final output Q. "
                f"Determine the exact binary value (0 or 1) of Q."
            ),
            "expected_answer": str(output),
            "tolerance": 0.0
        })
    return tasks


def generate_circuit_analysis_tasks(n: int = 20) -> list[dict]:
    """Generates passive analog circuit parameter calculations (RC cutoff & LC resonance)."""
    tasks = []
    r_values = [100, 220, 470, 1000, 2200, 4700, 10000]       # Ohms
    c_values_nf = [10, 22, 47, 100, 220, 470]                  # nanoFarads
    l_values_mh = [1, 2.2, 4.7, 10, 22, 47]                    # milliHenries

    for i in range(1, n + 1):
        if i % 2 == 1:
            # RC Low-Pass Filter -3dB Cutoff Frequency: f_c = 1 / (2 * pi * R * C)
            r = random.choice(r_values)
            c_nf = random.choice(c_values_nf)
            c = c_nf * 1e-9
            fc_hz = 1.0 / (2.0 * math.pi * r * c)
            expected = round(fc_hz, 2)

            tasks.append({
                "id": f"circuit_{i:02d}",
                "domain": "analog_circuits",
                "task": (
                    f"Calculate the -3 dB cutoff frequency (in Hz, rounded to 2 decimal places) "
                    f"of a first-order RC low-pass filter with R = {r} Ohms and C = {c_nf} nF."
                ),
                "expected_answer": f"{expected:.2f}",
                "tolerance": 0.5  # Allows minor pi rounding differences
            })
        else:
            # LC Resonant Frequency: f_0 = 1 / (2 * pi * sqrt(L * C))
            l_mh = random.choice(l_values_mh)
            c_nf = random.choice(c_values_nf)
            l_h = l_mh * 1e-3
            c_f = c_nf * 1e-9
            f0_khz = (1.0 / (2.0 * math.pi * math.sqrt(l_h * c_f))) / 1000.0
            expected = round(f0_khz, 2)

            tasks.append({
                "id": f"circuit_{i:02d}",
                "domain": "analog_circuits",
                "task": (
                    f"Calculate the resonant frequency (in kHz, rounded to 2 decimal places) "
                    f"of an ideal series LC circuit with L = {l_mh} mH and C = {c_nf} nF."
                ),
                "expected_answer": f"{expected:.2f}",
                "tolerance": 0.05
            })
    return tasks


def generate_cpp_trace_tasks(n: int = 20) -> list[dict]:
    """Generates C++ bitwise and pointer/array arithmetic output-prediction tasks."""
    tasks = []
    for i in range(1, n + 1):
        base = random.randint(5, 25)
        shift = random.randint(1, 3)
        mask = random.choice([0x0F, 0x1F, 0x3F])
        val = ((base << shift) ^ mask) & 0xFF

        tasks.append({
            "id": f"cpp_{i:02d}",
            "domain": "cpp_execution",
            "task": (
                f"Determine the exact integer printed to stdout by the following C++ snippet:\n"
                f"int x = {base};\n"
                f"int y = ((x << {shift}) ^ 0x{mask:02X}) & 0xFF;\n"
                f"std::cout << y;"
            ),
            "expected_answer": str(val),
            "tolerance": 0.0
        })
    return tasks


def main():
    random.seed(RANDOM_SEED)
    dataset = (
        generate_digital_logic_tasks(20)
        + generate_circuit_analysis_tasks(20)
        + generate_cpp_trace_tasks(20)
    )

    output_path = Path("data/raw/dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} deterministic benchmark tasks -> {output_path}")


if __name__ == "__main__":
    main()