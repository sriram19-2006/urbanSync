from src.simulator import TrafficSimulator


def main() -> None:
    simulator = TrafficSimulator()
    results = simulator.run(steps=12)

    print("Smart Traffic Signal Simulator")
    print("=" * 32)
    for item in results:
        emergency = " emergency override" if item["emergency_active"] else ""
        print(
            f"Step {item['step']:02d}: green={item['green_lane']} "
            f"counts={item['lane_counts']}{emergency}"
        )


if __name__ == "__main__":
    main()
