def calculate_transmission(distance_km, attenuation_db_km=0.2):
    """Return the fraction of photons that survive the fibre."""
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")
    if attenuation_db_km < 0:
        raise ValueError("Attenuation cannot be negative.")

    total_loss_db = distance_km * attenuation_db_km
    return 10 ** (-total_loss_db / 10)


def simulate_communication(
    distance_km,
    photons_sent=1_000_000_000,
    attenuation_db_km=0.2,
    detector_efficiency=0.8,
):
    """Calculate expected photon arrivals and detections."""
    if photons_sent < 0:
        raise ValueError("Photon count cannot be negative.")
    if not 0 <= detector_efficiency <= 1:
        raise ValueError("Detector efficiency must be between 0 and 1.")

    transmission = calculate_transmission(distance_km, attenuation_db_km)
    detection_probability = transmission * detector_efficiency

    return {
        "Distance": f"{distance_km:g} km",
        "Photons sent": f"{photons_sent:,.0f}",
        "Fibre loss": f"{distance_km * attenuation_db_km:.2f} dB",
        "Transmission probability": f"{transmission:.6g}",
        "Detection probability": f"{detection_probability:.6g}",
        "Expected arrivals": f"{photons_sent * transmission:,.0f}",
        "Expected detections": f"{photons_sent * detection_probability:,.0f}",
        "Photons lost in fibre": f"{(1 - transmission) * 100:.2f}%",
    }


if __name__ == "__main__":
    results = simulate_communication(
        distance_km=100,
        photons_sent=1_000_000_000,
        attenuation_db_km=0.2,
        detector_efficiency=0.8,
    )

    print("QUANTUM COMMUNICATION SIMULATOR")
    print("-" * 34)
    for label, value in results.items():
        print(f"{label}: {value}")
