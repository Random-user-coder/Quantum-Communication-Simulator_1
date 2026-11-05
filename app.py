"""
Q-COMM - Photon transmission simulator
======================================

A scientific workstation for the optical-fibre photon transmission model in
``physics.py``.

Sections
--------
Overview            context, live configuration, optical schematic, entry point
Simulator           controls, live results, link budget, photon journey,
                    parameter relationships, CSV export
Physics             equations, worked example, assumptions, limitations
Research Assistant  full workspace for the local rule-based assistant

The floating assistant (bottom-right) is available on every section except the
dedicated Research Assistant page, and shares one conversation with it.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import inspect
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Built-in physics model (no separate physics.py required)
# -----------------------------------------------------------------------------

import math


def calculate_transmission(distance_km, attenuation_db_km=0.2):
    """
    Calculate the probability that a photon survives
    transmission through an optical fibre.

    T = 10^(-alpha * L / 10)

    alpha = Fibre attenuation (dB/km)
    L = Fibre distance (km)
    """

    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    if attenuation_db_km < 0:
        raise ValueError("Attenuation cannot be negative.")

    total_loss_db = distance_km * attenuation_db_km

    transmission_probability = 10 ** (-total_loss_db / 10)

    return transmission_probability


def simulate_communication(
    distance_km,
    photons_sent=1_000_000_000,
    attenuation_db_km=0.2,
    detector_efficiency=0.8
):
    """
    Simulate photon transmission and detection.

    Detection probability = Transmission probability * Detector efficiency
    """

    if photons_sent < 0:
        raise ValueError("Photon count cannot be negative.")

    if not 0 <= detector_efficiency <= 1:
        raise ValueError("Detector efficiency must be between 0 and 1.")

    transmission = calculate_transmission(
        distance_km,
        attenuation_db_km
    )

    # Expected number of photons arriving after fibre transmission
    expected_arrivals = photons_sent * transmission

    # Expected number of photons successfully detected
    expected_detections = expected_arrivals * detector_efficiency

    # Probability of detecting one photon sent
    detection_probability = transmission * detector_efficiency

    # Percentage of photons lost inside the fibre
    photon_loss_percentage = (1 - transmission) * 100

    return {
        "Distance (km)": distance_km,
        "Photons sent": photons_sent,
        "Total fibre loss (dB)": distance_km * attenuation_db_km,
        "Transmission probability": transmission,
        "Detection probability": detection_probability,
        "Expected arrivals": expected_arrivals,
        "Expected detections": expected_detections,
        "Photon loss (%)": photon_loss_percentage
    }


# -----------------------------------------------------------------------------
# Built-in local chatbot (no separate assistant.py or API key required)
# -----------------------------------------------------------------------------

LEVELS = ("Beginner", "Intermediate", "Research")
DEFAULT_LEVEL = "Intermediate"

ASSISTANT_NOTE = (
    "Rule-based assistant: answers come from a fixed knowledge base and your "
    "current simulation values. No external model, no API key, no internet."
)

SUGGESTED_QUESTIONS = (
    "Why is my detection probability so low?",
    "What happens if I double the distance?",
    "What does 0.2 dB/km mean?",
    "Explain my photon journey diagram",
    "How can I improve the simulated detection probability?",
    "What are the limitations of this model?",
    "What is the difference between arrivals and detections?",
    "How does this relate to QKD?",
)

HELP_TOPICS = (
    ("Photon transmission", "What is transmission probability?"),
    ("Fibre attenuation", "What does 0.2 dB/km mean?"),
    ("Detector efficiency", "What is detector efficiency?"),
    ("Distance dependence", "Why does distance reduce transmission?"),
    ("Arrivals vs detections", "What is the difference between arrivals and detections?"),
    ("Photon loss", "Why are some photons lost?"),
    ("Equations", "Show me the equations"),
    ("Graphs", "Explain my photon journey diagram"),
    ("Parameters", "What do the parameters mean?"),
    ("Assumptions", "What are the limitations of this model?"),
    ("QKD concepts", "How does this relate to QKD?"),
)


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------

def _sig(x: float, digits: int = 4) -> str:
    """Compact significant-figure formatting that never shows float noise."""
    if x == 0:
        return "0"
    if abs(x) >= 1e-2:
        return f"{x:.{digits}g}"
    return f"{x:.{digits - 1}e}"


def prob_text(p: float) -> str:
    return _sig(p)


def plain_prob(p: float) -> str:
    """Decimal notation for beginners; e-notation only when it is unavoidable."""
    if p <= 0:
        return "0"
    if p >= 1e-4:
        return f"{p:.4g}"
    return f"{p:.2e}"


def pct_text(p: float, digits: int = 3) -> str:
    """A probability rendered as a percentage, switching to e-notation when tiny."""
    value = p * 100.0
    if value == 0:
        return "0%"
    if value < 1e-2:
        return f"{value:.2e}%"
    return f"{value:.{digits}g}%"


def count_text(n: float) -> str:
    if n == 0:
        return "0"
    if n < 1e-2 or n >= 1e9:
        return f"{n:.3e}"
    if n >= 1000:
        return f"{n:,.0f}"
    if n >= 1:
        return f"{n:,.2f}"
    return f"{n:.4f}"


def one_in(p: float) -> str:
    """'1 in N' phrasing for a per-photon probability."""
    if p <= 0:
        return "no photons"
    n = 1.0 / p
    if n >= 1e6:
        return f"1 in {n:.2e}"
    if n >= 10:
        return f"1 in {n:,.0f}"
    return f"{n:.2f} in 1"


def db_text(x: float, digits: int = 2) -> str:
    if math.isinf(x):
        return "infinite"
    return f"{x:.{digits}f} dB"


# ----------------------------------------------------------------------------
# Simulation context
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationContext:
    """Everything the assistant may quote, derived by the physics module."""

    distance_km: float
    attenuation_db_km: float
    detector_efficiency: float
    photons_sent: float
    transmission: float
    detection_probability: float
    expected_arrivals: float
    expected_detections: float
    photon_loss_percent: float
    fibre_loss_db: float

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        distance_km: float,
        attenuation_db_km: float,
        detector_efficiency: float,
        photons_sent: float,
    ) -> "SimulationContext":
        results = simulate_communication(
            distance_km=distance_km,
            photons_sent=photons_sent,
            attenuation_db_km=attenuation_db_km,
            detector_efficiency=detector_efficiency,
        )
        return cls(
            distance_km=float(distance_km),
            attenuation_db_km=float(attenuation_db_km),
            detector_efficiency=float(detector_efficiency),
            photons_sent=float(photons_sent),
            transmission=results["Transmission probability"],
            detection_probability=results["Detection probability"],
            expected_arrivals=results["Expected arrivals"],
            expected_detections=results["Expected detections"],
            photon_loss_percent=results["Photon loss (%)"],
            fibre_loss_db=results["Total fibre loss (dB)"],
        )

    # -- derived quantities ------------------------------------------------
    @property
    def detector_loss_db(self) -> float:
        eta = self.detector_efficiency
        if eta <= 0:
            return float("inf")
        return -10.0 * math.log10(eta)

    @property
    def total_loss_db(self) -> float:
        return self.fibre_loss_db + self.detector_loss_db

    @property
    def dominating_stage(self) -> str:
        return "fibre" if self.fibre_loss_db >= self.detector_loss_db else "detector"

    @property
    def span_10db_km(self) -> float:
        """Distance over which the fibre costs 10 dB (a factor of ten)."""
        if self.attenuation_db_km <= 0:
            return float("inf")
        return 10.0 / self.attenuation_db_km

    @property
    def span_3db_km(self) -> float:
        if self.attenuation_db_km <= 0:
            return float("inf")
        return 3.0 / self.attenuation_db_km

    @property
    def arrivals_per_1000(self) -> float:
        return 1000.0 * self.transmission

    @property
    def detections_per_1000(self) -> float:
        return 1000.0 * self.detection_probability

    @property
    def poisson_sigma(self) -> float:
        return math.sqrt(self.expected_detections) if self.expected_detections > 0 else 0.0


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def _pick(level: str, basic: str, standard: str, advanced: str) -> str:
    if level == "Beginner":
        return basic
    if level == "Research":
        return advanced
    return standard


def _has(question: str, *words: str) -> bool:
    return any(word in question for word in words)


def _clean(text: str) -> str:
    """Trim blank lines without destroying the leading indent of code blocks."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _budget_rows(ctx: SimulationContext) -> str:
    return (
        f"| Stage | Loss | Remaining fraction |\n"
        f"| --- | --- | --- |\n"
        f"| Fibre ({ctx.distance_km:g} km x {ctx.attenuation_db_km:g} dB/km) | "
        f"{ctx.fibre_loss_db:.2f} dB | {prob_text(ctx.transmission)} |\n"
        f"| Detector (eta = {ctx.detector_efficiency:.0%}) | "
        f"{ctx.detector_loss_db:.2f} dB | {ctx.detector_efficiency:.2f} |\n"
        f"| **Total link** | **{ctx.total_loss_db:.2f} dB** | "
        f"**{prob_text(ctx.detection_probability)}** |"
    )


# ----------------------------------------------------------------------------
# Topics
# ----------------------------------------------------------------------------

def _topic_what_if(question: str, ctx: SimulationContext, level: str):
    """Handle 'what if I double the distance / halve the attenuation' style asks."""
    factor = None
    label = None

    if re.search(r"\b(double|doubling|twice|2x|two times)\b", question):
        factor, label = 2.0, "doubled"
    elif re.search(r"\b(triple|tripling|three times|3x)\b", question):
        factor, label = 3.0, "tripled"
    elif re.search(r"\b(ten times|10x|order of magnitude)\b", question):
        factor, label = 10.0, "increased tenfold"
    elif re.search(r"\b(half|halve|halving|halved|50\s*%)\b", question):
        factor, label = 0.5, "halved"

    perfect_detector = bool(re.search(r"\b(perfect|ideal|100\s*%)\b", question)) and _has(
        question, "detector", "efficiency"
    )

    if factor is None and not perfect_detector:
        return None

    target = None
    if _has(question, "distance", "length", "span", "km"):
        target = "distance"
    elif _has(question, "attenuation", "db/km", "db per km", "fibre quality"):
        target = "attenuation"
    elif _has(question, "efficiency", "detector", "eta"):
        target = "detector"
    if target is None:
        return None

    distance = ctx.distance_km
    attenuation = ctx.attenuation_db_km
    efficiency = ctx.detector_efficiency
    headline = ""

    if target == "distance":
        distance = max(ctx.distance_km * factor, 0.1)
        headline = f"If the fibre span were {label} to {distance:g} km"
    elif target == "attenuation":
        attenuation = ctx.attenuation_db_km * factor
        headline = f"If the attenuation were {label} to {attenuation:g} dB/km"
    else:
        efficiency = 1.0 if perfect_detector else min(ctx.detector_efficiency * factor, 1.0)
        headline = f"If the detector efficiency were {efficiency:.0%}"

    alt = SimulationContext.build(distance, attenuation, efficiency, ctx.photons_sent)

    ratio = (
        alt.detection_probability / ctx.detection_probability
        if ctx.detection_probability > 0
        else float("nan")
    )
    extra_db = alt.total_loss_db - ctx.total_loss_db

    change = "improves" if ratio > 1 else "reduces"
    change_amount = f"x{ratio:.3g}" if ratio > 1 else f"to {ratio * 100:.3g}% of its present value"

    basic = _clean(f"""
{headline}:

| | now | what-if |
| --- | --- | --- |
| Detection probability | {plain_prob(ctx.detection_probability)} | {plain_prob(alt.detection_probability)} |
| Expected detections | {count_text(ctx.expected_detections)} | {count_text(alt.expected_detections)} |

That {change} the chance of registering a photon {change_amount}, using the
same {ctx.photons_sent:,.0f} input photons.

The reason: {abs(extra_db):.2f} dB {'more' if extra_db > 0 else 'less'} loss, and every
10 dB means ten times fewer photons.
""")

    if target == "distance":
        mechanism = (
            "Distance sits in the exponent: multiplying the span multiplies the dB loss by the\n"
            "same factor and raises the survival probability to a power."
        )
    elif target == "attenuation":
        mechanism = (
            "Attenuation also sits in the exponent: multiplying alpha multiplies the total dB\n"
            "loss by that factor, so the penalty compounds with the length of the span."
        )
    else:
        mechanism = (
            "Detector efficiency does not sit in the exponent: it scales the arriving photon\n"
            "number linearly, so it shifts the dB budget by a constant amount."
        )

    standard = _clean(f"""
{headline}:

| Quantity | Now | What-if |
| --- | --- | --- |
| Fibre loss | {ctx.fibre_loss_db:.2f} dB | {alt.fibre_loss_db:.2f} dB |
| Transmission T | {prob_text(ctx.transmission)} | {prob_text(alt.transmission)} |
| Detection probability | {plain_prob(ctx.detection_probability)} | {plain_prob(alt.detection_probability)} |
| Expected detections | {count_text(ctx.expected_detections)} | {count_text(alt.expected_detections)} |

Change in total link loss: {extra_db:+.2f} dB, so the detection probability
becomes {ratio:.4g} times its present value.

{mechanism}
""")

    if target == "distance":
        insight = (
            f"Doubling the span costs an extra {ctx.fibre_loss_db:.2f} dB, which is exactly the "
            f"loss of the original span. In general T(2L) = T(L)^2, so every doubling squares the "
            f"survival probability. This is why the ladder 50 km -> 100 km -> 200 km moves you from "
            f"1 in 10 to 1 in 100 to 1 in 10,000 photons."
        )
    elif target == "attenuation":
        insight = (
            f"Attenuation enters the exponent linearly: a factor of {factor:g} on alpha multiplies "
            f"the dB loss by {factor:g} and raises the transmission to the power {factor:g}. "
            f"Because the loss is additive in dB, the improvement compounds with distance - "
            f"which is why ultra-low-loss fibre matters most on long spans."
        )
    else:
        insight = (
            "Detector efficiency is a plain multiplier on the arriving flux and never touches the "
            "fibre term, so it enters the link budget as a fixed dB offset "
            f"({ctx.detector_loss_db:.2f} dB at eta = {ctx.detector_efficiency:.0%}). "
            "It improves the count rate linearly while the fibre term improves it exponentially "
            "with distance."
        )

    advanced = standard + "\n\n" + _clean(f"""
Research note
-------------
{insight}

Log-domain check: the detection probability is
  10^(-(alpha L + 10 log10(1/eta)) / 10)
so the two stages are additive in dB and multiplicative in linear terms.
With {count_text(ctx.expected_detections)} expected detections at present, the
shot-noise standard deviation is about {count_text(ctx.poisson_sigma)}
(sqrt of the mean), i.e. a relative uncertainty near
{(100 / math.sqrt(ctx.expected_detections)):.3g}% if the run were repeated.
""")

    return _pick(level, basic, standard, advanced)


def _topic_improve(ctx: SimulationContext, level: str) -> str:
    better_fibre = SimulationContext.build(
        ctx.distance_km, max(ctx.attenuation_db_km - 0.04, 0.05), ctx.detector_efficiency, ctx.photons_sent
    )
    better_detector = SimulationContext.build(
        ctx.distance_km, ctx.attenuation_db_km, min(ctx.detector_efficiency + 0.15, 1.0), ctx.photons_sent
    )
    shorter = SimulationContext.build(
        max(ctx.distance_km * 0.5, 0.1), ctx.attenuation_db_km, ctx.detector_efficiency, ctx.photons_sent
    )

    gain_f = better_fibre.detection_probability / ctx.detection_probability if ctx.detection_probability else 0
    gain_d = better_detector.detection_probability / ctx.detection_probability if ctx.detection_probability else 0
    gain_s = shorter.detection_probability / ctx.detection_probability if ctx.detection_probability else 0

    basic = _clean(f"""
Three things limit your detection probability, and all three can be changed.

1. Shorten the fibre. Light is lost a little on every kilometre.
   Halving the span to {shorter.distance_km:g} km would raise the chance of a
   detection by about {gain_s:.3g} times in this model.

2. Use better fibre. Lower dB/km means less loss per kilometre.
   Going from {ctx.attenuation_db_km:g} to {better_fibre.attenuation_db_km:g} dB/km would raise it by
   about {gain_f:.3g} times.

3. Use a better detector. A detector that registers more of the photons that
   arrive helps directly. Going from {ctx.detector_efficiency:.0%} to
   {better_detector.detector_efficiency:.0%} efficiency would raise it by about {gain_d:.3g} times.

Right now {ctx.dominating_stage} loss is the bigger problem
({db_text(ctx.fibre_loss_db)} of fibre loss against {db_text(ctx.detector_loss_db)}
of detector loss), so that is where effort pays off most.
""")

    standard = basic + "\n\n" + _clean(f"""
What the numbers mean
---------------------
Sensitivity of the model, holding the other two parameters fixed:

| Lever | Change | Detection probability |
| --- | --- | --- |
| Distance | {ctx.distance_km:g} km to {shorter.distance_km:g} km | x{gain_s:.3g} |
| Attenuation | {ctx.attenuation_db_km:g} to {better_fibre.attenuation_db_km:g} dB/km | x{gain_f:.3g} |
| Detector efficiency | {ctx.detector_efficiency:.0%} to {better_detector.detector_efficiency:.0%} | x{gain_d:.3g} |

Two practical limits this model does not include:

* Optical amplifiers can boost classical signals but cannot copy an unknown
  single photon, so they do not repair single-photon links.
* Detector efficiency is usually traded against dark counts: the most efficient
  single-photon detectors at 1550 nm are also the ones that demand cryogenics.
""")

    advanced = standard + "\n\n" + _clean(f"""
Research framing
----------------
The link budget is additive in dB:

    accepted = alpha L + 10 log10(1/eta)
    currently {ctx.fibre_loss_db:.2f} + {ctx.detector_loss_db:.2f} = {ctx.total_loss_db:.2f} dB

Improving alpha by d(alpha) buys 10 log10(e) * L * d(alpha) dB, so the payoff is
linear in span length: fibre quality dominates on long links, detector quality
dominates on short ones. The crossover for this configuration sits where
alpha * L equals 10 log10(1/eta), i.e. about
{(10 * math.log10(1 / ctx.detector_efficiency)) / ctx.attenuation_db_km:.0f} km.
Beyond that point, every dB saved in the fibre is worth more than a dB saved
in the detector.

For quantum links, photon-number-preserving gain is impossible (no-cloning),
so the only structural fixes are lower alpha, higher eta, shorter spans, or
trusted-node / quantum-memory architectures.
""")

    return _pick(level, basic, standard, advanced)


def _topic_diagnosis(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
Your detection probability is {plain_prob(ctx.detection_probability)}. In plain terms:
out of every 1,000 photons you send, about {ctx.detections_per_1000:.3g} are
registered by the detector.

Two stages reduce the number:

1. The fibre. {ctx.distance_km:g} km at {ctx.attenuation_db_km:g} dB/km removes
   {ctx.fibre_loss_db:.1f} dB of light, so only about {one_in(ctx.transmission)}
   photons reaches the far end.

2. The detector. A detector at {ctx.detector_efficiency:.0%} efficiency registers
   {ctx.detector_efficiency:.0%} of the photons that reach it.

The two stages multiply, so the effects compound:

    {plain_prob(ctx.transmission)}  x  {ctx.detector_efficiency:.2f}  =  {plain_prob(ctx.detection_probability)}

With {db_text(ctx.fibre_loss_db)} of fibre loss against {db_text(ctx.detector_loss_db)}
of detector loss, {ctx.dominating_stage} loss is the dominant limitation here.
""")

    standard = _clean(f"""
Detection probability is {prob_text(ctx.detection_probability)}, i.e. {pct_text(ctx.detection_probability)}.

The model applies two factors in series:

    P_detection = T x eta
    T = 10^(-alpha L / 10) = 10^(-({ctx.attenuation_db_km:g} x {ctx.distance_km:g}) / 10)
      = {prob_text(ctx.transmission)}

so:

    P_detection = {prob_text(ctx.transmission)} x {ctx.detector_efficiency:.2f}
                = {prob_text(ctx.detection_probability)}

Loss budget:

{_budget_rows(ctx)}

With {ctx.photons_sent:,.0f} photons sent, that predicts
{count_text(ctx.expected_arrivals)} arrivals and
{count_text(ctx.expected_detections)} detections.

Why it feels "low": the fibre term is exponential in distance. Each 10 dB costs
a factor of 10 in photon number, and {ctx.distance_km:g} km already costs
{ctx.fibre_loss_db:.1f} dB ({ctx.fibre_loss_db / 10:.2f} factors of ten).
""")

    advanced = standard + "\n\n" + _clean(f"""
Research reading
----------------
The link is limited by propagation, not by the detector: fibre contributes
{ctx.fibre_loss_db:.2f} dB and the detector only {ctx.detector_loss_db:.2f} dB.
Restating in the exponent,

    P = 10^(-(alpha L + 10 log10(1/eta)) / 10)
      = 10^(-({ctx.fibre_loss_db:.2f} + {ctx.detector_loss_db:.2f}) / 10)
      = {prob_text(ctx.detection_probability)}

Distinguish the quantities the interface reports:

* T = {prob_text(ctx.transmission)} - photons that survive the fibre (all losses inside the glass).
* eta = {ctx.detector_efficiency:.2f} - fraction of those arriving photons that produce a click.
* P_detection = {prob_text(ctx.detection_probability)} - joint probability for one sent photon.
* N_detected = {count_text(ctx.expected_detections)} - expectation for {ctx.photons_sent:,.0f} sent photons.

Because P << 1, repeated single-photon trials are well described by a Poisson
process with mean mu = {count_text(ctx.expected_detections)} and sigma = sqrt(mu)
= {count_text(ctx.poisson_sigma)}. Relative shot noise is therefore about
{(100 / math.sqrt(ctx.expected_detections)) if ctx.expected_detections > 0 else float('nan'):.3g}%,
which is the resolution limit of any measurement made at this operating point.

Nothing here is a measurement: the values are the exact expectation of the
attenuation model under the stated assumptions.
""")

    return _pick(level, basic, standard, advanced)


def _topic_attenuation(ctx: SimulationContext, level: str) -> str:
    a = ctx.attenuation_db_km
    sample = []
    for d in (1, 10, 50, 100):
        share = 10 ** (-a * d / 10)
        sample.append((d, share))
    rows = "\n".join(f"    after {d:>3} km: {pct_text(s, 3):>9} of the light remains" for d, s in sample)

    basic = _clean(f"""
Attenuation is how much light the fibre absorbs and scatters on every
kilometre. It is measured in decibels per kilometre (dB/km).

Your value is {a:g} dB/km, which means:

{rows}

So a signal does not fade politely - it collapses exponentially. At your
distance of {ctx.distance_km:g} km the total is {ctx.fibre_loss_db:.1f} dB and
{pct_text(ctx.transmission)} of the photons survive.
""")

    standard = basic + "\n\n" + _clean(f"""
The model
---------
Loss in decibels accumulates additively along the fibre:

    L_dB = alpha x L = {a:g} x {ctx.distance_km:g} = {ctx.fibre_loss_db:.2f} dB

and the surviving fraction converts back with

    T = 10^(-L_dB / 10) = 10^(-{ctx.fibre_loss_db:.2f} / 10) = {prob_text(ctx.transmission)}

Useful anchors: 3 dB halves the photon number, 10 dB divides it by ten,
20 dB divides it by a hundred. A {a:g} dB/km fibre therefore costs 10 dB every
{ctx.span_10db_km:g} km, and 3 dB every {ctx.span_3db_km:g} km.

For context, standard single-mode fibre at 1550 nm is specified near
0.20 dB/km, and ultra-low-loss fibre reaches roughly 0.16-0.18 dB/km.
1310 nm systems sit nearer 0.35 dB/km.
""")

    advanced = standard + "\n\n" + _clean(f"""
Physical origin
---------------
In silica the 1550 nm window is a compromise between two intrinsic terms:
Rayleigh scattering, which falls as 1/lambda^4, and the infrared absorption
tail of the glass, which rises with wavelength. The residual OH- absorption
peaks near 1383 nm are the reason the low-loss band is split around it.

The dB scale is used because each metre of fibre multiplies the surviving
amplitude independently, so logarithms turn a chain of multiplications into a
sum:

    T_total = prod_i 10^(-alpha_i L_i / 10)  =>  L_dB,total = sum_i alpha_i L_i

Consequences for the model: alpha is treated as uniform and linear along the
span. Real links add connector and splice losses - typically a few tenths of a
dB per event - which this simulator does not include. Stimulated Brillouin and
Raman scattering, four-wave mixing and polarisation-mode dispersion are also
outside the model; they matter for power levels well above single-photon
regime.
""")

    return _pick(level, basic, standard, advanced)


def _topic_detector(ctx: SimulationContext, level: str) -> str:
    eta = ctx.detector_efficiency

    basic = _clean(f"""
Detector efficiency is the share of arriving photons the detector actually
registers. Yours is set to {eta:.0%}: for every 100 photons that reach it, about
{eta * 100:.0f} produce a detection.

Those missing photons were not lost in the fibre - they arrived, but the
detector did not respond. That is a separate loss from the fibre loss.

In the model: detection probability = transmission x efficiency
    {prob_text(ctx.transmission)} x {eta:.2f} = {prob_text(ctx.detection_probability)}
""")

    standard = basic + "\n\n" + _clean(f"""
Numbers for your configuration
------------------------------
    arriving photons        {count_text(ctx.expected_arrivals)}
    registered by detector  {count_text(ctx.expected_detections)}
    missed by detector      {count_text(ctx.expected_arrivals - ctx.expected_detections)}

Expressed as a loss, eta = {eta:.0%} corresponds to {ctx.detector_loss_db:.2f} dB:

    L_eta = -10 log10(eta) = -10 log10({eta:.2f}) = {ctx.detector_loss_db:.2f} dB

Compare that with the fibre term of {ctx.fibre_loss_db:.2f} dB. Detector
inefficiency is usually the smaller of the two, but it is the easier one to
engineer because it does not grow with distance.
""")

    advanced = standard + "\n\n" + _clean(f"""
Research notes
--------------
eta bundles several physical effects into one number: fibre-to-detector
coupling, the probability the absorber produces an excitation, and the
probability that excitation is converted into a registered click. It is not a
property of the fibre and never enters T.

At 1550 nm the usual trade-off is:

* InGaAs/InP SPADs: room temperature or modest cooling, efficiency roughly
  10-30%, dark counts and afterpulsing limit the usable rate.
* Superconducting nanowire detectors (SNSPD): efficiency above 90% has been
  demonstrated, at the cost of cryogenics and low saturation counts.

Because P_detection = T x eta, the detector term is a constant multiplicative
factor, so it is additive in dB and independent of span length. That is why
detector improvements help most exactly where the fibre term is small.

This model assumes a single scalar eta with no dark counts, no dead time, no
afterpulsing and no timing jitter. In a real experiment the recorded count rate
would be eta x flux plus a dark-count floor, and at very long spans that floor
- not the fibre - sets the limit.
""")

    return _pick(level, basic, standard, advanced)


def _topic_distance(ctx: SimulationContext, level: str) -> str:
    T = ctx.transmission
    double_T = T ** 2

    basic = _clean(f"""
Distance hurts because loss accumulates on every kilometre. It is not a straight
line - the number of surviving photons falls away much faster than the distance
grows.

At your {ctx.distance_km:g} km, {ctx.fibre_loss_db:.1f} dB is lost and about
{one_in(T)} photons gets through.

Double the distance and you double the dB, which squares the survival
probability: {prob_text(T)} becomes {prob_text(double_T)}.
""")

    standard = _clean(f"""
Transmission falls exponentially with distance:

    T(L) = 10^(-alpha L / 10)
    T({ctx.distance_km:g} km) = 10^(-{ctx.attenuation_db_km:g} x {ctx.distance_km:g} / 10)
              = {prob_text(T)}

Three consequences:

1. Two spans in series multiply, so T doubles the length -> T^2:
   T(2L) = T(L)^2 = {prob_text(double_T)}.
2. Each 10 dB of loss divides the photon count by ten. At {ctx.attenuation_db_km:g} dB/km
   that happens every {ctx.span_10db_km:g} km.
3. Each 3 dB halves it, i.e. every {ctx.span_3db_km:g} km.

This is why "just add more fibre" fails: the penalty is multiplicative in
photons even though it is additive in decibels.
""")

    advanced = standard + "\n\n" + _clean(f"""
Formal view
-----------
Write the survival probability per unit length as p = 10^(-alpha/10) per km.
Then T(L) = p^L, an exponential decay with characteristic length

    1 / ln(1/p) = 10 / (alpha ln 10) ~ {10 / (ctx.attenuation_db_km * math.log(10)):.1f} km (1/e length)

and the engineering figures of merit

    L_10dB = 10 / alpha = {ctx.span_10db_km:.1f} km
    L_3dB  =  3 / alpha = {ctx.span_3db_km:.1f} km

Because L appears in the exponent, sensitivity dT/dL = -T alpha ln(10)/10 is
proportional to T itself: the link degrades gracefully at short spans and
catastrophically once T approaches the detection floor. For
{ctx.photons_sent:,.0f} input photons and a mean of one detection, the useful
ceiling sits near L such that N T eta ~ 1, i.e. roughly
{ctx.distance_km + 10 * math.log10(max(ctx.expected_detections, 1e-300)) / ctx.attenuation_db_km:.0f} km for this configuration.
""")

    return _pick(level, basic, standard, advanced)


def _topic_arrivals(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
"Arrivals" are the photons that survive the fibre and reach the detector.
"Detections" are the smaller number that the detector actually registers.

With {ctx.photons_sent:,.0f} photons sent:

    arrive at the detector   {count_text(ctx.expected_arrivals)}
    registered               {count_text(ctx.expected_detections)}

The difference ({count_text(ctx.expected_arrivals - ctx.expected_detections)})
is not lost in the glass - it reached the detector and was simply missed.
""")

    standard = basic + "\n\n" + _clean(f"""
    N_arrivals   = N_sent x T            = {ctx.photons_sent:,.0f} x {prob_text(ctx.transmission)} = {count_text(ctx.expected_arrivals)}
    N_detections = N_arrivals x eta      = {count_text(ctx.expected_arrivals)} x {ctx.detector_efficiency:.2f} = {count_text(ctx.expected_detections)}

Both are expected values for the whole ensemble. A single photon is detected
or it is not; the model predicts the average over many trials, never the
outcome of one.
""")

    advanced = standard + "\n\n" + _clean(f"""
Statistical structure
---------------------
Per sent photon the outcome is Bernoulli with p = {prob_text(ctx.detection_probability)}.
For N = {ctx.photons_sent:,.0f} independent trials the count is Binomial(N, p),
with mean {count_text(ctx.expected_detections)} and variance N p (1-p).
Since p << 1 the Poisson limit applies:

    sigma = sqrt(N p (1-p)) = {count_text(math.sqrt(ctx.expected_detections * (1 - ctx.detection_probability)) if ctx.expected_detections > 0 else 0)}

Relative fluctuation ~ {(100 / math.sqrt(ctx.expected_detections)) if ctx.expected_detections > 0 else float('nan'):.3g}%.
Two-stage attenuation preserves the statistics: thinning a Poisson stream by T
and then by eta yields a Poisson stream with mean reduced by T eta.
""")

    return _pick(level, basic, standard, advanced)


def _topic_detections(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
Expected detections is the average number of photons the detector will register
for your input.

You send {ctx.photons_sent:,.0f} photons, {ctx.detection_probability * 100:.4g}% of them are
registered on average, so the model predicts about
{count_text(ctx.expected_detections)} detections.

It is an average over many repeats, not a guaranteed count: real runs scatter
around it.
""")

    standard = basic + "\n\n" + _clean(f"""
    N_detected = N_sent x T x eta
               = {ctx.photons_sent:,.0f} x {prob_text(ctx.transmission)} x {ctx.detector_efficiency:.2f}
               = {count_text(ctx.expected_detections)}

Equivalent phrasing: one detection per {one_in(ctx.detection_probability)} photons sent.
The value scales linearly with the number of input photons - multiplying the
source rate by ten multiplies the detection count by ten - but exponentially
with distance, because T sits in the exponent.
""")

    advanced = standard + "\n\n" + _clean(f"""
Precision
---------
The expected count is exact arithmetic on the model, but the observable count is
random. For N p >> 1 the relative standard deviation is 1/sqrt(N p), here about
{(100 / math.sqrt(ctx.expected_detections)) if ctx.expected_detections > 0 else float('nan'):.3g}%.
Once N p drops below ~1 the notion of a "rate" stops being useful and you have
to reason in terms of detection probability per trial instead - which is exactly
the regime long-haul single-photon links operate in.
""")

    return _pick(level, basic, standard, advanced)


def _topic_photon_loss(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
Photons are lost for two quite different reasons, and it helps to keep them
apart.

1. Inside the fibre: the glass scatters a little light out of the core and
   absorbs a little. Over {ctx.distance_km:g} km that removes
   {ctx.photon_loss_percent:.4g}% of the photons, leaving {prob_text(ctx.transmission)}
   of them.

2. At the detector: a {ctx.detector_efficiency:.0%} efficient detector misses
   {100 * (1 - ctx.detector_efficiency):.0f}% of the photons that do arrive.

Only the first is fibre loss. The second is a detection loss.
""")

    standard = basic + "\n\n" + _clean(f"""
Breakdown for this run
----------------------
    sent                    {ctx.photons_sent:,.0f}
    lost in fibre           {count_text(ctx.photons_sent * (1 - ctx.transmission))}
    arrived                 {count_text(ctx.expected_arrivals)}
    missed by detector      {count_text(ctx.expected_arrivals - ctx.expected_detections)}
    detected                {count_text(ctx.expected_detections)}

The fraction lost in the glass is (1 - T) x 100 = {ctx.photon_loss_percent:.4f}%,
matching the "Photon loss (%)" metric. It depends only on alpha and L - the
detector cannot change it.
""")

    advanced = standard + "\n\n" + _clean(f"""
Mechanisms
----------
* Rayleigh scattering from density fluctuations frozen into the glass -
  the dominant intrinsic term near 1550 nm, scaling as 1/lambda^4.
* Infrared absorption by the silica network, rising with wavelength; the
  minimum between the two is the 1550 nm window.
* Residual OH- absorption peaks (notably near 1383 nm), now largely removed in
  low-water-peak fibre.
* Bend loss and cladding-mode leakage in real cable, not represented here.
* Connectors, splices and patch panels: each of order 0.1-0.5 dB. A link with
  ten such events can lose several dB that this uniform-alpha model attributes
  to nothing.

Detection loss is described by eta and is independent of span length, so on long
links fibre loss always wins; on short laboratory links detector efficiency
usually does.
""")

    return _pick(level, basic, standard, advanced)


def _topic_equations(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
The simulator uses three simple rules.

1. Photons left after the fibre:

       T = 10^(-(attenuation x distance) / 10)

2. Chance the detector registers a photon:

       detection probability = T x efficiency

3. How many detections to expect:

       expected detections = photons sent x detection probability

With your settings that is
{prob_text(ctx.transmission)} x {ctx.detector_efficiency:.2f} = {prob_text(ctx.detection_probability)},
so about {count_text(ctx.expected_detections)} detections from {ctx.photons_sent:,.0f} photons.
""")

    standard = _clean(f"""
    T          = 10^(-alpha L / 10)
    P_detect   = T x eta
    N_detected = N_sent x P_detect

    alpha  attenuation [dB/km]
    L      fibre length [km]
    eta    detector efficiency [0-1]

Solved for your configuration:

    T          = 10^(-{ctx.attenuation_db_km:g} x {ctx.distance_km:g} / 10) = {prob_text(ctx.transmission)}
    P_detect   = {prob_text(ctx.transmission)} x {ctx.detector_efficiency:.2f} = {prob_text(ctx.detection_probability)}
    N_detected = {ctx.photons_sent:,.0f} x {prob_text(ctx.detection_probability)} = {count_text(ctx.expected_detections)}

Conversions worth remembering:

    L_dB   = -10 log10(T)          T = 10^(-L_dB/10)
    3 dB   -> factor 2
    10 dB  -> factor 10
    20 dB  -> factor 100
""")

    advanced = standard + "\n\n" + _clean(f"""
Derivation
----------
For a uniform fibre the intensity decays as I(L) = I0 exp(-gamma L) with a
power attenuation coefficient gamma. Defining the decibel as a power ratio,

    alpha[dB/km] = (10 / L) log10(I0 / I(L))  =>  I(L)/I0 = 10^(-alpha L/10)

At the single-photon level the classical intensity ratio is the survival
probability of one photon, so T = 10^(-alpha L/10) is exact for a Poisson or
Fock state under a beam-splitter model of loss.

Detection is an independent binary filter, so the joint efficiency is the
product of the two transmission factors:

    P = T eta = 10^(-(alpha L + 10 log10(1/eta))/10)

That identity is the whole link budget: dB from propagation plus dB from the
detector, summed in the exponent. All numbers reported by Q-COMM follow from it;
no empirical fit or experimental data enters the calculation.
""")

    return _pick(level, basic, standard, advanced)


def _topic_graphs(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
There are three views of the same model.

* Photon journey (Sankey): shows where the photons go. A thick band leaves
  "sent", splits into "lost in fibre" and "arriving", then the arriving band
  splits again into "missed" and "detected". Your run:
  {count_text(ctx.expected_arrivals)} arrive and {count_text(ctx.expected_detections)} are detected.

* Detection probability vs distance: curves down steeply, then flattens on the
  logarithmic axis. That straight line is what exponential decay looks like
  when the axis is logarithmic.

* Detection probability vs attenuation and vs efficiency: shorter distance or a
  better detector moves the whole curve up.
""")

    standard = basic + "\n\n" + _clean(f"""
How to read the axes
--------------------
The probability axis is logarithmic. Straight lines therefore mean exponential
decay, and the slope carries information: for a decade of photon number you
need 10 dB, i.e. 10 / alpha = {ctx.span_10db_km:.0f} km at {ctx.attenuation_db_km:g} dB/km.

The marker on each curve is your current operating point
(L = {ctx.distance_km:g} km, alpha = {ctx.attenuation_db_km:g} dB/km,
eta = {ctx.detector_efficiency:.0%}, P = {prob_text(ctx.detection_probability)}).

The attenuation and efficiency curves are drawn with the other two parameters
held fixed, which is what "partial dependence" means: each chart answers "what
if only this one thing changed".
""")

    advanced = standard + "\n\n" + _clean(f"""
Interpreting the shapes
-----------------------
log10 P is affine in L with slope -alpha/10 = {-ctx.attenuation_db_km / 10:.4g} decades per km,
so any feature that looks like a straight line on the log plot is a pure
exponential. In alpha the relation is also exponential
(P ~ 10^(-alpha L/10)), while in eta it is exactly linear (P = T eta) - which is
why the efficiency chart is a straight line on linear axes and the others need a
log axis to stay readable.

The Sankey diagram is a conservation diagram for expected counts, not a
simulation of individual trajectories: band widths are N_sent, N_sent(1-T),
N_sent T, and so on. Because it plots expectations, it stays smooth even where
a real experiment would show 0 or 1 counts per run.
""")

    return _pick(level, basic, standard, advanced)


def _topic_parameters(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
Your four controls:

    Fibre distance       {ctx.distance_km:g} km      longer fibre -> fewer photons
    Attenuation          {ctx.attenuation_db_km:g} dB/km   loss per kilometre
    Detector efficiency  {ctx.detector_efficiency:.0%}       share of arrivals registered
    Photons sent         {ctx.photons_sent:,.0f}   size of the input pulse

Only the first three change the probability per photon. The fourth changes how
many counts you expect, without changing any probability.
""")

    standard = basic + "\n\n" + _clean(f"""
| Parameter | Effect on P_detection | Typical range |
| --- | --- | --- |
| Distance L | 10^(-alpha L / 10) | 1 - 1000 km |
| Attenuation alpha | 10^(-alpha L / 10) | 0.16 - 0.35 dB/km |
| Efficiency eta | proportional to eta | 10 - 95% |
| Photons N_sent | unchanged; counts scale with N | any |

At {ctx.attenuation_db_km:g} dB/km this link costs 10 dB every {ctx.span_10db_km:.0f} km, which is
the single most useful number for predicting the effect of distance.
""")

    advanced = standard + "\n\n" + _clean(f"""
Parameter sensitivity at this operating point
---------------------------------------------
    dP/dL      = -P alpha ln(10)/10 = {-ctx.detection_probability * ctx.attenuation_db_km * math.log(10) / 10:.4g} per km
    dP/dalpha  = -P L ln(10)/10    = {-ctx.detection_probability * ctx.distance_km * math.log(10) / 10:.4g} per dB/km
    dP/deta    =  T                = {prob_text(ctx.transmission)} per unit efficiency

Note the asymmetry: efficiency is the only linear control. The other two act
through the exponent, so their influence grows with the span you are trying to
bridge - which is precisely why the fibre specification dominates link design.
""")

    return _pick(level, basic, standard, advanced)


def _topic_limitations(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
This is a simplified model, and it is honest about that.

What it does include:
- fibre attenuation that grows with distance
- a detector that registers only a fraction of arriving photons

What it leaves out:
- background light and detector dark counts
- amplifiers, repeaters and quantum memories
- weather, turbulence and free-space links
- full QKD protocols and their security analysis
- random run-to-run fluctuations in the counts

So treat every number as a theoretical expectation, not a measurement.
""")

    standard = _clean(f"""
Model scope and assumptions
---------------------------
Included:
* uniform attenuation alpha over the whole span (no connectors, splices or bends)
* a single scalar detector efficiency eta
* ideal, noiseless, single-photon-number-resolving statistics via expectation values

Not included:
* dark counts, afterpulsing, dead time, timing jitter, background light
* optical amplifiers, repeaters, trusted nodes, quantum memories
* polarisation drift, dispersion, nonlinear effects
* atmospheric turbulence and free-space or satellite channels
* multi-photon emission, decoy states, error correction, privacy amplification
* any QKD protocol layer or secret-key rate

The outputs are therefore the deterministic expectation of the attenuation
model. A real experiment would add noise and systematic effects.
""")

    advanced = standard + "\n\n" + _clean(f"""
Where the model would break first
---------------------------------
The idealisation with the shortest shelf life is the absence of dark counts.
At the present setting the expected signal is
{count_text(ctx.expected_detections)} counts per {ctx.photons_sent:,.0f} photons; a real
detector contributes a dark-count floor of order 10-1000 counts per second,
which would dominate as soon as the signal rate falls below it. Restating the
same point in probability terms: the model still reports
{prob_text(ctx.detection_probability)} at {ctx.distance_km:g} km, while a physical
link would be noise-limited long before that value became small.

A second idealisation is treating alpha as uniform. Real spans add splices and
connectors (0.1-0.5 dB each); a link with twenty of them can carry several dB
that the model never charges for, which is precisely the margin a link budget
would have to reserve.

Third, eta is assumed constant. In practice it depends on count rate (dead time,
afterpulsing) and on polarisation, so a rate-dependent detector model is the
natural next step.

Finally, the simulator stops at the link layer. It does not implement BB84 or
any other protocol, so it makes no statement about secret-key rate, error
rates or security.
""")

    return _pick(level, basic, standard, advanced)


def _topic_qkd(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
QKD (quantum key distribution) uses single photons to build a shared secret key.
Because an unknown photon cannot be copied, an eavesdropper's attempt shows up
as errors in the key.

Loss is the central problem: the key rate shrinks as photons fail to arrive.
Your link keeps {plain_prob(ctx.detection_probability)} of the photons, which is why
long-distance quantum links need either very good fibre, very good detectors,
or intermediate nodes.

Q-COMM does not implement a QKD protocol. It only models the transmission that
any such protocol would have to work with.
""")

    standard = basic + "\n\n" + _clean(f"""
Where your numbers would enter a protocol
-----------------------------------------
In BB84 with an ideal single-photon source, the sifted key rate per pulse is of
order (1/2) x T x eta - the factor 1/2 from basis reconciliation, the rest from
exactly the quantities this simulator computes. At your settings:

    T x eta = {prob_text(ctx.detection_probability)} per photon

so a {ctx.photons_sent:,.0f}-photon burst yields roughly
{count_text(ctx.expected_detections)} detection events, and only a fraction of
those survive error correction and privacy amplification.

Missing from this model: basis choice, measurement, error estimation, decoy
states, error correction, privacy amplification and any security proof.
""")

    advanced = standard + "\n\n" + _clean(f"""
Research reading
----------------
For an ideal single-photon BB84 implementation the asymptotic secret-key rate is
proportional to the overall transmission, R ~ O(T eta) per pulse after sifting,
so it inherits exactly the exponential distance penalty computed here. Practical
sources are weak coherent pulses, where the multi-photon fraction forces decoy
states and the rate picks up an additional e^(-mu) factor; photon-number
splitting then bounds the secure distance further.

Two consequences follow directly from the physics in this module:

* Because loss is multiplicative, each {ctx.span_10db_km:.0f} km of fibre costs one decade in
  key rate at {ctx.attenuation_db_km:g} dB/km. Every 3 dB of extra loss halves the rate.
* No-cloning forbids amplification of an unknown quantum state, so classical
  optical amplifiers cannot extend a QKD link; the options are lower alpha,
  higher eta, or trusted nodes / quantum repeaters.

Q-COMM stops at the link layer by design: it computes transmission and detection
probability, not key rate, and makes no security claim.
""")

    return _pick(level, basic, standard, advanced)


def _topic_overview(ctx: SimulationContext, level: str) -> str:
    basic = _clean(f"""
Q-COMM models what happens to photons sent down an optical fibre.

Light leaks out of the glass as it travels, and the detector does not register
every photon that arrives. Q-COMM multiplies those two effects to give the
probability that a photon you send is actually detected.

Right now: {ctx.distance_km:g} km of fibre at {ctx.attenuation_db_km:g} dB/km, a
{ctx.detector_efficiency:.0%} efficient detector, so
{plain_prob(ctx.detection_probability)} of the photons are detected -
about {count_text(ctx.expected_detections)} out of {ctx.photons_sent:,.0f}.
""")

    standard = basic + "\n\n" + _clean(f"""
How it is put together
----------------------
    fibre        T = 10^(-alpha L / 10)                      = {prob_text(ctx.transmission)}
    detector     eta                                         = {ctx.detector_efficiency:.2f}
    link         P = T eta                                   = {prob_text(ctx.detection_probability)}
    counts       N = N_sent x P                              = {count_text(ctx.expected_detections)}

The interface has four sections: Overview (context and schematic), Simulator
(controls, live results, photon journey, relationships, export), Physics
(equations, assumptions, limitations) and Research Assistant.
""")

    advanced = standard + "\n\n" + _clean(f"""
Positioning
-----------
Q-COMM is an analytical link-budget tool for single-photon transmission. It
computes the expectation of a beam-splitter loss model followed by a binary
detection filter; it does not simulate trajectories, detector physics or any
protocol layer. That narrowness is deliberate: every number it reports can be
checked by hand from the three equations, which is what makes it useful as a
teaching and first-pass design instrument.
""")

    return _pick(level, basic, standard, advanced)


def _topic_help(ctx: SimulationContext, level: str) -> str:
    lines = "\n".join(f"* {label} - e.g. \"{example}\"" for label, example in HELP_TOPICS)
    return _clean(f"""
I answer from a fixed knowledge base and your current simulation values.

{lines}

I also handle hypotheticals such as "what if I double the distance" or
"what if the detector were perfect", and I can quote any current value.

I am not a language model: if a question falls outside these topics I will say
so rather than guess.
""")


def _fallback(ctx: SimulationContext, level: str) -> str:
    if level == "Beginner":
        return _clean(f"""
I could not match that to something I know about.

I can explain, in simple terms:
- what the fibre does to the photons
- what detector efficiency means
- why distance matters so much
- where the photons are lost
- what the numbers on screen mean

Try one of the suggested questions, or ask "what can you answer?".
""")

    return _clean(f"""
I could not match that question to an entry in my knowledge base, so I will not
guess at an answer.

Topics I can cover: transmission probability, attenuation and dB arithmetic,
detector efficiency, the distance law, expected arrivals and detections, photon
loss mechanisms, the three equations, the graphs and Sankey diagram, the
parameters, model assumptions and limitations, and introductory QKD.

Ask "what can you answer?" for examples. For a numeric hypothetical, try
"what if I double the distance" or "what if the detector were perfect".
""")


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------

def local_assistant(
    question: str,
    ctx: SimulationContext,
    level: str = DEFAULT_LEVEL,
) -> str:
    """Answer a question using the rule base and the current simulation context."""
    if level not in LEVELS:
        level = DEFAULT_LEVEL
    if not question or not question.strip():
        return _topic_help(ctx, level)

    q = " ".join(question.lower().split())

    # 1. Hypotheticals and "how do I improve" style asks come first because they
    #    contain many of the topic words used further down.
    what_if = _topic_what_if(q, ctx, level)
    if what_if:
        return what_if

    if _has(q, "improve", "better", "optimise", "optimize", "boost", "increase the",
            "increase my", "maximise", "maximize", "stronger signal"):
        return _topic_improve(ctx, level)

    if _has(q, "why is my", "why are my", "so low", "explain my result",
            "my result", "current result", "interpret my", "analyse my", "analyze my",
            "what does this mean for my"):
        return _topic_diagnosis(ctx, level)

    if _has(q, "help", "what can you", "what do you know", "topics", "capabilities"):
        return _topic_help(ctx, level)

    if _has(q, "qkd", "quantum key", "bb84", "entanglement", "qubit", "eavesdrop",
            "secret key", "cryptograph"):
        return _topic_qkd(ctx, level)

    if _has(q, "equation", "formula", "maths", "math", "derive", "derivation", "calculate"):
        return _topic_equations(ctx, level)

    if _has(q, "graph", "chart", "plot", "sankey", "diagram", "journey", "axis", "log scale"):
        return _topic_graphs(ctx, level)

    if _has(q, "arrive", "arrival", "reach the detector", "reaches the detector"):
        return _topic_arrivals(ctx, level)

    if _has(q, "detection", "detections", "count", "counts", "registered"):
        return _topic_detections(ctx, level)

    if _has(q, "lost", "loss", "why are some photons", "disappear", "vanish"):
        return _topic_photon_loss(ctx, level)

    if _has(q, "attenuation", "db/km", "db per km", "decibel", "db mean", "loss per km"):
        return _topic_attenuation(ctx, level)

    if _has(q, "efficiency", "detector", "eta", "spad", "snspd"):
        return _topic_detector(ctx, level)

    if _has(q, "distance", "length", "span", "km", "longer", "shorter"):
        return _topic_distance(ctx, level)

    if _has(q, "parameter", "slider", "control", "setting", "configure", "what do the numbers"):
        return _topic_parameters(ctx, level)

    if _has(q, "limitation", "assumption", "assume", "realistic", "accuracy", "valid",
            "not include", "missing", "dark count"):
        return _topic_limitations(ctx, level)

    if _has(q, "transmission", "survive", "survival", "through the fibre", "probability"):
        return _topic_diagnosis(ctx, level)

    if _has(q, "q-comm", "qcomm", "simulator", "what is this", "overview", "about"):
        return _topic_overview(ctx, level)

    return _fallback(ctx, level)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Q-COMM | Photon Transmission Simulator",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="auto",
)

PAGES = ("Overview", "Simulator", "Physics", "Research Assistant")

DEFAULTS = {
    "distance": 100,
    "attenuation": 0.20,
    "efficiency": 80,
    "log_photons": 9.0,
}

LIMITS = {
    "distance": (1, 1000, 1),
    "attenuation": (0.05, 0.60, 0.01),
    "efficiency": (10, 100, 1),
    "log_photons": (3.0, 15.0, 0.5),
}

FIBRE_PRESETS = {
    "Standard single-mode, 1550 nm": 0.20,
    "Ultra-low-loss, 1550 nm": 0.16,
    "Single-mode, 1310 nm": 0.35,
}

DETECTOR_PRESETS = {
    "InGaAs SPAD, 1550 nm": 25,
    "SNSPD, 1550 nm": 90,
    "Silicon SPAD, 800 nm": 70,
}

C = {
    "bg": "#0a0f15",
    "panel": "#0f1720",
    "panel_2": "#121c27",
    "line": "#1f2d3b",
    "line_soft": "#18232f",
    "text": "#dde6ee",
    "muted": "#8697a9",
    "cyan": "#4cc6d8",
    "amber": "#d99b45",
    "green": "#5fae83",
    "trace": "#3f5a70",
    "grid": "rgba(255,255,255,0.055)",
}

SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

# ----------------------------------------------------------------------------
# Version-tolerant layout helpers
# ----------------------------------------------------------------------------


def _fit(func) -> dict:
    """Return the full-width argument supported by this Streamlit version."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


CHART_FIT = _fit(st.plotly_chart)
TABLE_FIT = _fit(st.dataframe)
BTN_FIT = _fit(st.button)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

CSS = """
<style>
:root {
  --qc-bg:#0a0f15; --qc-panel:#0f1720; --qc-panel-2:#121c27;
  --qc-line:#1f2d3b; --qc-line-soft:#18232f;
  --qc-text:#dde6ee; --qc-muted:#8697a9;
  --qc-cyan:#4cc6d8; --qc-amber:#d99b45; --qc-green:#5fae83;
}
.stApp { background:var(--qc-bg); color:var(--qc-text); }
[data-testid="stHeader"] { background:transparent; height:2.1rem; }
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stToolbar"] { visibility:hidden; }
.block-container { padding:1.0rem 2.2rem 2.4rem; max-width:1500px; }

/* typography: scoped so KaTeX and icon ligature fonts keep their own families */
html, body, .stApp { font-family:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif; }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label, button, input, textarea, select, p, li, td, th {
  font-family:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
}
[data-testid="stMarkdownContainer"] h1, .stMarkdown h1 {
  font-size:1.58rem !important; font-weight:600 !important; letter-spacing:-0.022em;
  margin:0 0 .2rem !important; padding:0 !important; line-height:1.25;
}
[data-testid="stMarkdownContainer"] h2, .stMarkdown h2 {
  font-size:1.08rem !important; font-weight:600 !important; letter-spacing:-0.01em;
  margin:0 0 .55rem !important; padding:0 !important;
}
[data-testid="stMarkdownContainer"] h3, .stMarkdown h3 {
  font-size:.93rem !important; font-weight:600 !important; letter-spacing:-0.005em;
  margin:0 0 .4rem !important; padding:0 !important;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size:.89rem; line-height:1.58; }
/* restore icon + math fonts explicitly */
[data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"],
span[class*="material-symbols"], span[class*="MaterialSymbols"] {
  font-family:"Material Symbols Rounded","Material Icons" !important;
}
.katex, .katex * { font-family:KaTeX_Main,KaTeX_Math,"Times New Roman",serif !important; }
.katex .mathnormal, .katex .mord.mathnormal { font-family:KaTeX_Math,KaTeX_Main,serif !important; }
[data-testid="stMarkdownContainer"] p { margin-bottom:.55rem; }

/* sidebar */
[data-testid="stSidebar"] { background:#0b1219; border-right:1px solid var(--qc-line); }
[data-testid="stSidebar"] > div:first-child { padding-top:.4rem; }
[data-testid="stSidebar"] hr { border-color:var(--qc-line); margin:.9rem 0; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.1rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:.44rem .55rem; border-radius:5px; border:1px solid transparent;
  transition:background .12s, border-color .12s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#111b25; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:#11202a; border-color:#274553;
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size:.86rem; }

/* brand */
.qc-brand { font-size:.94rem; font-weight:700; letter-spacing:.18em; color:var(--qc-text); }
.qc-brand span { color:var(--qc-cyan); }
.qc-brand-sub { color:var(--qc-muted); font-size:.72rem; margin-top:.15rem; letter-spacing:.02em; }
.qc-side-label {
  color:var(--qc-muted); font-size:.64rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; margin:.15rem 0 .35rem;
}

/* typographic furniture */
.qc-eyebrow {
  color:var(--qc-cyan); font-size:.65rem; font-weight:600; letter-spacing:.17em;
  text-transform:uppercase; margin-bottom:.4rem;
}
.qc-sub { color:var(--qc-muted); font-size:.84rem; line-height:1.5; max-width:78ch; }
.qc-rule { height:1px; background:var(--qc-line); margin:.95rem 0 1.15rem; }
.qc-foot { color:#6b7c8e; font-size:.72rem; line-height:1.5; }

/* metric cards */
.qc-metric {
  background:var(--qc-panel); border:1px solid var(--qc-line); border-left:2px solid #26384a;
  border-radius:6px; padding:.68rem .8rem .62rem; min-height:5.3rem;
}
.qc-metric-label {
  color:var(--qc-muted); font-size:.66rem; font-weight:600; letter-spacing:.09em;
  text-transform:uppercase;
}
.qc-metric-value {
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:1.16rem; font-weight:500; color:var(--qc-text);
  margin-top:.42rem; letter-spacing:-0.02em;
}
.qc-metric-foot { color:var(--qc-muted); font-size:.69rem; margin-top:.28rem; }

/* schematic */
.qc-schematic { display:flex; align-items:stretch; gap:0; }
.qc-node {
  flex:1 1 0; width:0; background:#0d151f; border:1px solid var(--qc-line);
  border-radius:6px; padding:.65rem .75rem .6rem; min-width:0;
}
.qc-node-title {
  font-size:.63rem; font-weight:600; letter-spacing:.13em; text-transform:uppercase;
  color:var(--qc-muted);
}
.qc-node-value {
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.98rem; color:var(--qc-text); margin-top:.3rem;
}
.qc-node-sub { font-size:.71rem; color:var(--qc-muted); margin-top:.12rem; }
.qc-link {
  flex:0 0 76px; display:flex; flex-direction:column; align-items:center;
  justify-content:center; gap:.28rem; padding:0 .2rem;
}
.qc-bar {
  width:100%; height:1px; position:relative;
  background:linear-gradient(90deg,#22364a,var(--qc-cyan));
}
.qc-bar::after {
  content:""; position:absolute; right:-1px; top:-3px;
  border-left:5px solid var(--qc-cyan);
  border-top:3.5px solid transparent; border-bottom:3.5px solid transparent;
}
.qc-link span {
  font-size:.63rem; color:#9fb3c6; font-family:ui-monospace,SFMono-Regular,monospace;
  white-space:nowrap; letter-spacing:-0.01em;
}

/* panels */
.qc-panel {
  background:var(--qc-panel); border:1px solid var(--qc-line); border-radius:6px;
  padding:.8rem .9rem .7rem;
}
.qc-key { display:flex; justify-content:space-between; gap:.8rem; padding:.3rem 0; border-bottom:1px solid var(--qc-line-soft); }
.qc-key:last-child { border-bottom:none; }
.qc-key-label { color:var(--qc-muted); font-size:.79rem; }
.qc-key-value {
  color:var(--qc-text); font-size:.79rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--qc-line); }
.stTabs [data-baseweb="tab"] {
  height:34px; padding:0; background:transparent; color:var(--qc-muted);
  font-size:.82rem; letter-spacing:.01em;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--qc-text); }
.stTabs [aria-selected="true"] { color:var(--qc-cyan); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--qc-cyan); height:1px; }
.stTabs [data-baseweb="tab-border"] { display:none; }

/* controls */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius:5px; border:1px solid var(--qc-line); background:#111b25; color:var(--qc-text);
  font-size:.8rem; font-weight:500; padding:.32rem .75rem; min-height:2.05rem;
  transition:border-color .12s, color .12s, background .12s;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  border-color:var(--qc-cyan); color:var(--qc-cyan); background:#0f1d26;
}
.stButton > button:focus:not(:active) { border-color:var(--qc-cyan); color:var(--qc-cyan); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background:#0e1721 !important; border-color:var(--qc-line) !important; border-radius:5px;
}
[data-baseweb="input"] input { font-size:.84rem; }
[data-testid="stWidgetLabel"] p { font-size:.8rem; color:var(--qc-muted); }
[data-testid="stExpander"] details { border:1px solid var(--qc-line); border-radius:6px; background:#0d151f; }
[data-testid="stExpander"] summary { font-size:.82rem; }
[data-testid="stExpander"] summary p { font-size:.82rem; }
[data-testid="stDataFrame"] { border:1px solid var(--qc-line); border-radius:6px; }
[data-testid="stVegaLiteChart"] { border:1px solid var(--qc-line); border-radius:6px; }

/* chat */
.qc-msg-user {
  background:#14202c; border:1px solid #22384a; border-radius:6px;
  padding:.45rem .6rem; margin:.15rem 0 .55rem auto; max-width:88%;
  font-size:.84rem; line-height:1.5; color:var(--qc-text); width:fit-content;
}
[class*="st-key-qcmsg"] {
  border-left:2px solid #22384a; padding:.05rem 0 .05rem .65rem; margin-bottom:.45rem;
}
[class*="st-key-qcmsg"] p { font-size:.84rem; line-height:1.52; margin-bottom:.45rem; }
[class*="st-key-qcmsg"] li { font-size:.84rem; line-height:1.5; }
[class*="st-key-qcmsg"] table { font-size:.78rem; }
[class*="st-key-qcmsg"] h1, [class*="st-key-qcmsg"] h2, [class*="st-key-qcmsg"] h3 { font-size:.88rem; }

/* floating assistant */
.st-key-qc_launcher { position:fixed; right:22px; bottom:22px; z-index:100001; width:auto; }
.st-key-qc_launcher button {
  border-radius:999px; padding:.5rem 1.05rem; background:#0f1720;
  border:1px solid #2b3f52; box-shadow:0 10px 30px rgba(0,0,0,.5);
  font-size:.82rem;
}
.st-key-qc_launcher button:hover { border-color:var(--qc-cyan); }
.st-key-qc_panel {
  position:fixed; right:22px; bottom:22px; z-index:100000;
  width:392px; max-width:calc(100vw - 28px);
  background:#0c141d; border:1px solid #263b4d; border-radius:9px;
  padding:.7rem .8rem .55rem; box-shadow:0 26px 68px rgba(0,0,0,.62);
}
.st-key-qc_mini {
  position:fixed; right:22px; bottom:22px; z-index:100000; width:266px;
  background:#0c141d; border:1px solid #263b4d; border-radius:9px;
  padding:.35rem .5rem; box-shadow:0 20px 50px rgba(0,0,0,.6);
}
.qc-panel-title { font-size:.72rem; font-weight:600; letter-spacing:.13em; text-transform:uppercase; color:var(--qc-text); }
.qc-panel-sub { font-size:.64rem; color:var(--qc-muted); letter-spacing:.03em; margin-top:.1rem; }
.st-key-qc_panel [data-testid="stVerticalBlock"] { gap:.35rem; }
.st-key-qc_panel .stButton > button { padding:.1rem .3rem; min-height:1.6rem; font-size:.78rem; }
.st-key-qc_mini .stButton > button { padding:.05rem .25rem; min-height:1.5rem; font-size:.75rem; }
.st-key-qc_panel [data-testid="stForm"] { border:none; padding:0; }
.qc-hint { color:#6b7c8e; font-size:.66rem; line-height:1.45; margin-top:.1rem; }

/* refinements */
[data-testid="stMarkdownContainer"] h3 { margin-top:1.25rem !important; }
[data-testid="stMarkdownContainer"] h3:first-child { margin-top:0 !important; }
.katex { font-size:1.04em; }
.katex-display { margin:.45rem 0 .6rem; }
[data-testid="stMarkdownContainer"] pre {
  background:#0b141d !important; border:1px solid var(--qc-line-soft); border-radius:5px;
  padding:.5rem .65rem;
}
[data-testid="stMarkdownContainer"] code { font-size:.8rem; color:#a8c6d8; }
[data-testid="stMarkdownContainer"] table { font-size:.82rem; }
[data-testid="stMarkdownContainer"] th { color:var(--qc-muted); font-weight:600; }
[data-testid="stExpander"] { margin-bottom:.35rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--qc-line) !important; }
.st-key-qc_panel { width:420px; }
.st-key-qc_mini { width:302px; }
.qc-link span { color:#9fb3c6; }

[data-testid="stPlotlyChart"] {
  border:1px solid var(--qc-line); border-radius:6px; background:var(--qc-panel);
  padding:.25rem .3rem .1rem;
}
[data-testid="stPlotlyChart"] .modebar { display:none !important; }
.qc-link span { font-size:.64rem; color:#9fb3c6; }

.st-key-qc_panel { max-height:calc(100vh - 44px); overflow-y:auto; }
.st-key-qc_panel [data-testid="stHorizontalBlock"] { gap:.35rem; }
.qc-panel-head { display:flex; }

@media (max-width: 780px) {
  .block-container { padding:.9rem 1rem 2rem; }
  .qc-schematic { flex-direction:column; gap:.5rem; }
  .qc-link { display:none; }
  .qc-node { width:auto; }
  .qc-metric { min-height:auto; }
  .st-key-qc_panel { width:calc(100vw - 20px); right:10px; bottom:10px; }
  .st-key-qc_mini { right:10px; bottom:10px; }
  .st-key-qc_launcher { right:10px; bottom:10px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State and parameter helpers
# ----------------------------------------------------------------------------


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("assistant_open", False)
    ss.setdefault("assistant_min", False)
    ss.setdefault("assistant_max", False)
    ss.setdefault("assistant_level", DEFAULT_LEVEL)
    ss.setdefault("assistant_messages", [])
    for key, value in DEFAULTS.items():
        ss.setdefault(f"p_{key}", value)


def apply_pending_reset() -> None:
    """Reset parameters before any widget is created in this run."""
    if not st.session_state.pop("_do_reset", False):
        return
    for key, value in DEFAULTS.items():
        _set_param(key, value)
    for key in ("fibre_preset", "detector_preset"):
        st.session_state.pop(key, None)


def _apply_fibre_preset() -> None:
    _set_param("attenuation", FIBRE_PRESETS[st.session_state["fibre_preset"]])


def _apply_detector_preset() -> None:
    _set_param("efficiency", DETECTOR_PRESETS[st.session_state["detector_preset"]])


def _set_param(key: str, value: float) -> None:
    """Write one parameter to the model and to both linked widgets."""
    ss = st.session_state
    ss[f"p_{key}"] = value
    ss[f"qc_s_{key}"] = value
    ss[f"qc_n_{key}"] = value


def linked_param(key: str, label: str, unit: str, integer: bool = False, help_text: str = "") -> float:
    """Slider plus exact numeric field, both bound to one session value."""
    ss = st.session_state
    lo, hi, step = LIMITS[key]
    base = f"p_{key}"
    slider_key, number_key = f"qc_s_{key}", f"qc_n_{key}"

    if slider_key not in ss:
        ss[slider_key] = ss[base]
    if number_key not in ss:
        ss[number_key] = ss[base]

    def from_slider() -> None:
        _set_param(key, ss[slider_key])

    def from_number() -> None:
        _set_param(key, ss[number_key])

    slider_col, number_col = st.columns([1.42, 0.5], gap="small")

    with slider_col:
        if integer:
            st.slider(
                label, int(lo), int(hi), step=int(step), key=slider_key,
                on_change=from_slider, help=help_text,
            )
        else:
            st.slider(
                label, float(lo), float(hi), step=float(step), key=slider_key,
                on_change=from_slider, help=help_text,
            )

    with number_col:
        if integer:
            st.number_input(
                f"exact {key}", min_value=int(lo), max_value=int(hi), step=int(step),
                key=number_key, on_change=from_number, label_visibility="collapsed",
            )
        else:
            st.number_input(
                f"exact {key}", min_value=float(lo), max_value=float(hi), step=float(step),
                format="%.2f", key=number_key, on_change=from_number,
                label_visibility="collapsed",
            )

    return float(ss[base])

# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------


def sci(value: float, digits: int = 2) -> str:
    """1.00 x 10^7 style formatting for counts and probabilities."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else "—"
    exponent = int(math.floor(math.log10(abs(value))))
    if exponent == 0:
        return f"{value:.{digits}f}"
    mantissa = value / 10 ** exponent
    return f"{mantissa:.{digits}f} x 10{str(exponent).translate(SUP)}"


def plain(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if 1e-3 <= abs(value) < 1e5:
        return f"{value:.{digits}g}"
    return sci(value)


def count_short(value: float) -> str:
    if value >= 1e5 or (value and value < 1e-2):
        return sci(value)
    return f"{value:,.0f}" if value >= 1 else f"{value:.3g}"


def percent(value: float, digits: int = 3) -> str:
    pct = value * 100
    if pct == 0:
        return "0%"
    if pct < 1e-2:
        return sci(pct, 2) + "%"
    return f"{pct:.{digits}g}%"


def latex_number(value: float, digits: int = 3) -> str:
    """LaTeX-safe number so small values do not render as 'e-10'."""
    if value == 0:
        return "0"
    if 1e-3 <= abs(value) < 1e4:
        return f"{value:.4g}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return rf"{mantissa:.{digits}f}\times10^{{{exponent}}}"


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="qc-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def page_head(title: str, subtitle: str, tag: str) -> None:
    eyebrow(tag)
    st.markdown(f"<h1>{html.escape(title)}</h1>", unsafe_allow_html=True)
    st.markdown(f'<div class="qc-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, foot: str = "", accent: str | None = None, tip: str = "") -> None:
    border = f"border-left:2px solid {accent};" if accent else ""
    title = f' title="{html.escape(tip)}"' if tip else ""
    st.markdown(
        f'<div class="qc-metric" style="{border}"{title}>'
        f'<div class="qc-metric-label">{html.escape(label)}</div>'
        f'<div class="qc-metric-value">{html.escape(value)}</div>'
        f'<div class="qc-metric-foot">{html.escape(foot)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def key_value_rows(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="qc-key"><span class="qc-key-label">{html.escape(k)}</span>'
        f'<span class="qc-key-value">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    return f'<div class="qc-panel">{body}</div>'


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------


def style_figure(fig: go.Figure, height: int = 330, title: str = "", x_title: str = "",
                 y_title: str = "", log_y: bool = False, legend: bool = True) -> go.Figure:
    axis = dict(
        gridcolor=C["grid"],
        zeroline=False,
        linecolor=C["line"],
        tickfont=dict(size=10.5, color=C["muted"]),
        title=dict(font=dict(size=11, color=C["muted"]), standoff=10),
    )
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["muted"]),
        height=height,
        margin=dict(l=8, r=12, t=42 if (title or legend) else 12, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1721", bordercolor=C["line"], font=dict(size=11, color=C["text"])),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        xaxis={**axis, "title": dict(text=x_title, font=dict(size=11, color=C["muted"]), standoff=10)},
        yaxis={**axis, "title": dict(text=y_title, font=dict(size=11, color=C["muted"]), standoff=8)},
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=12, color=C["text"]), x=0, xanchor="left",
            y=0.98, yanchor="top", pad=dict(l=0),
        )
    fig.update_layout(**layout)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="power", showgrid=True)
    return fig

def fig_link_budget(ctx: SimulationContext) -> go.Figure:
    fibre = -ctx.fibre_loss_db
    detector = -ctx.detector_loss_db
    total = fibre + detector
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Launched", "Fibre", "Detector", "At detector"],
            y=[0, fibre, detector, 0],
            text=[f"0 dB", f"{fibre:.2f} dB", f"{detector:.2f} dB", f"{total:.2f} dB"],
            textposition="outside",
            textfont=dict(size=10.5, color=C["muted"]),
            connector=dict(line=dict(color="#2b3e50", width=1)),
            increasing=dict(marker=dict(color=C["green"])),
            decreasing=dict(marker=dict(color=C["amber"])),
            totals=dict(marker=dict(color=C["cyan"])),
            hovertemplate="%{x}: %{y:.2f} dB<extra></extra>",
            width=0.55,
        )
    )
    style_figure(fig, height=300, title="Optical link budget", y_title="Power (dB)", legend=False)
    fig.update_yaxes(range=[min(total, fibre) - 4, 3])
    return fig


def fig_vs_distance(ctx: SimulationContext) -> go.Figure:
    span = max(400.0, ctx.distance_km * 1.5)
    distances = np.linspace(1.0, span, 240)
    transmission = [calculate_transmission(d, ctx.attenuation_db_km) for d in distances]
    detection = [t * ctx.detector_efficiency for t in transmission]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances, y=detection, name="Detection probability",
            line=dict(color=C["cyan"], width=2),
            hovertemplate="%{x:.0f} km<br>P = %{y:.3e}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ctx.distance_km], y=[ctx.detection_probability], mode="markers",
            name="Operating point",
            marker=dict(color=C["amber"], size=9, symbol="diamond", line=dict(color=C["bg"], width=1)),
            hovertemplate="current setting<br>%{x:.0f} km<br>P = %{y:.3e}<extra></extra>",
        )
    )
    style_figure(fig, height=340, title=f"Detection probability against fibre length (eta = {ctx.detector_efficiency:.0%})",
                 x_title="Fibre length (km)", y_title="Probability", log_y=True)
    return fig


def fig_vs_attenuation(ctx: SimulationContext) -> go.Figure:
    values = np.linspace(LIMITS["attenuation"][0], LIMITS["attenuation"][1], 200)
    detection = [calculate_transmission(ctx.distance_km, a) * ctx.detector_efficiency for a in values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values, y=detection, name="Detection probability",
            line=dict(color=C["cyan"], width=2),
            hovertemplate="%{x:.2f} dB/km<br>P = %{y:.3e}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ctx.attenuation_db_km], y=[ctx.detection_probability], mode="markers",
            name="Operating point",
            marker=dict(color=C["amber"], size=9, symbol="diamond", line=dict(color=C["bg"], width=1)),
            hovertemplate="current setting<br>%{x:.2f} dB/km<br>P = %{y:.3e}<extra></extra>",
        )
    )
    style_figure(fig, height=320, title=f"Against attenuation, L = {ctx.distance_km:g} km",
                 x_title="Attenuation (dB/km)", y_title="Probability", log_y=True)
    return fig


def fig_vs_efficiency(ctx: SimulationContext) -> go.Figure:
    values = np.linspace(0.05, 1.0, 120)
    detection = [ctx.transmission * eta for eta in values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values * 100, y=detection, name="Detection probability",
            line=dict(color=C["cyan"], width=2),
            hovertemplate="%{x:.0f}%<br>P = %{y:.3e}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ctx.detector_efficiency * 100], y=[ctx.detection_probability], mode="markers",
            name="Operating point",
            marker=dict(color=C["amber"], size=9, symbol="diamond", line=dict(color=C["bg"], width=1)),
            hovertemplate="current setting<br>%{x:.0f}%<br>P = %{y:.3e}<extra></extra>",
        )
    )
    style_figure(fig, height=320, title="Against detector efficiency (linear response)",
                 x_title="Detector efficiency (%)", y_title="Probability")
    return fig


def fig_sankey(ctx: SimulationContext) -> go.Figure:
    sent = ctx.photons_sent
    lost_fibre = sent * (1 - ctx.transmission)
    arrivals = ctx.expected_arrivals
    missed = arrivals * (1 - ctx.detector_efficiency)
    detected = ctx.expected_detections

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=[
                    f"Sent<br>{sci(sent)}",
                    f"Lost in fibre<br>{sci(lost_fibre)}",
                    f"Arrived<br>{sci(arrivals)}",
                    f"Not registered<br>{sci(missed)}",
                    f"Detected<br>{sci(detected)}",
                ],
                x=[0.02, 0.46, 0.46, 0.98, 0.98],
                y=[0.5, 0.22, 0.84, 0.90, 0.72],
                color=[C["cyan"], C["amber"], C["cyan"], C["amber"], C["green"]],
                pad=30,
                thickness=15,
                line=dict(color=C["bg"], width=1),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=[0, 0, 2, 2],
                target=[1, 2, 3, 4],
                value=[max(lost_fibre, 0), max(arrivals, 0), max(missed, 0), max(detected, 0)],
                color=[
                    "rgba(217,155,69,0.26)",
                    "rgba(76,198,216,0.24)",
                    "rgba(217,155,69,0.20)",
                    "rgba(95,174,131,0.38)",
                ],
                hovertemplate="%{source.label} to %{target.label}<br>%{value:.3e} photons<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["muted"]),
        height=372,
        margin=dict(l=6, r=150, t=16, b=10),
        hoverlabel=dict(bgcolor="#0d1721", bordercolor=C["line"], font=dict(size=11, color=C["text"])),
    )
    return fig

def metrics_row(ctx: SimulationContext) -> None:
    cols = st.columns(4, gap="small")
    with cols[0]:
        metric_card("Transmission probability", f"{ctx.transmission:.4g}",
                    "photons surviving the fibre", C["cyan"],
                    "T = 10^(-alpha L / 10): the share of photons that cross the full fibre span.")
    with cols[1]:
        metric_card("Detection probability", f"{ctx.detection_probability:.4g}",
                    "per photon launched", C["cyan"],
                    "P = T x eta: fibre survival multiplied by detector efficiency.")
    with cols[2]:
        metric_card("Expected detections", count_short(ctx.expected_detections),
                    f"from {count_short(ctx.photons_sent)} launched", C["green"],
                    "N_sent x P. An expectation for the ensemble, not a single-run count.")
    with cols[3]:
        metric_card("Fibre photon loss", f"{ctx.photon_loss_percent:.2f}%",
                    f"{ctx.fibre_loss_db:.2f} dB accumulated", C["amber"],
                    "(1 - T) x 100. Loss inside the glass only; detector misses are separate.")


def schematic(ctx: SimulationContext) -> None:
    st.markdown(
        f"""
<div class="qc-schematic">
  <div class="qc-node">
    <div class="qc-node-title">Transmitter</div>
    <div class="qc-node-value">{html.escape(count_short(ctx.photons_sent))}</div>
    <div class="qc-node-sub">photons launched</div>
  </div>
  <div class="qc-link"><div class="qc-bar"></div><span>T = {html.escape(plain(ctx.transmission))}</span></div>
  <div class="qc-node">
    <div class="qc-node-title">Optical fibre</div>
    <div class="qc-node-value">{ctx.distance_km:g} km</div>
    <div class="qc-node-sub">{ctx.attenuation_db_km:g} dB/km | {ctx.fibre_loss_db:.1f} dB</div>
  </div>
  <div class="qc-link"><div class="qc-bar"></div><span>eta = {ctx.detector_efficiency:.2f}</span></div>
  <div class="qc-node">
    <div class="qc-node-title">Detector</div>
    <div class="qc-node-value">{html.escape(count_short(ctx.expected_detections))}</div>
    <div class="qc-node-sub">{ctx.detector_efficiency:.0%} efficiency</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_overview(ctx: SimulationContext) -> None:
    page_head(
        "Photon transmission simulator",
        "Analytical model of single-photon transmission through uniform optical fibre: "
        "attenuation along the span, then a detector that registers only part of what arrives.",
        "Q-COMM / optical link analysis",
    )

    metrics_row(ctx)
    st.write("")
    schematic(ctx)
    st.write("")

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown("### What the model computes")
        st.markdown(
            "Two independent stages reduce the number of usable photons. The fibre attenuates "
            "exponentially with distance, and the detector converts only a fraction of the "
            "arrivals into clicks. Both are multiplied into a single link efficiency."
        )
        st.latex(r"T = 10^{-\alpha L / 10} \qquad P_{\mathrm{det}} = T\,\eta \qquad N_{\mathrm{det}} = N_{\mathrm{sent}}P_{\mathrm{det}}")
        st.markdown(
            '<div class="qc-sub">Every number in this interface follows from those three lines. '
            "No empirical fit, no measured dataset, no protocol layer.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.expander("Reading the headline numbers"):
            st.markdown(
                f"""
**Transmission probability** {plain(ctx.transmission)} - the share of photons that survive
{ctx.distance_km:g} km of fibre. At {ctx.attenuation_db_km:g} dB/km the span costs
{ctx.fibre_loss_db:.2f} dB, so about {plain(ctx.transmission)} of the photons remain.

**Detection probability** {plain(ctx.detection_probability)} - the share that survive the fibre
*and* register in a detector running at {ctx.detector_efficiency:.0%} efficiency.

**Expected detections** {count_short(ctx.expected_detections)} - the same probability applied to
{count_short(ctx.photons_sent)} launched photons. This is an average over many repeats, not a
prediction for one run.

**Fibre photon loss** {ctx.photon_loss_percent:.4g}% - loss inside the glass only. Photons missed
by the detector arrived safely and are counted separately.
"""
            )

    with right:
        st.markdown("### Current configuration")
        st.markdown(
            key_value_rows(
                [
                    ("Fibre length", f"{ctx.distance_km:g} km"),
                    ("Attenuation", f"{ctx.attenuation_db_km:g} dB/km"),
                    ("Total fibre loss", f"{ctx.fibre_loss_db:.2f} dB"),
                    ("Detector efficiency", f"{ctx.detector_efficiency:.0%}"),
                    ("Detector loss", f"{ctx.detector_loss_db:.2f} dB"),
                    ("Photons launched", count_short(ctx.photons_sent)),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Open the simulator", key="go_simulator", type="primary", **BTN_FIT):
            st.session_state["_goto"] = "Simulator"
            st.rerun()
        st.markdown(
            '<div class="qc-hint">Controls live in the sidebar; the simulator adds the link '
            "budget, the photon journey and parameter relationships.</div>",
            unsafe_allow_html=True,
        )


def page_simulator(ctx: SimulationContext) -> None:
    page_head(
        "Simulator",
        "Adjust the experiment in the sidebar. Results, the photon journey and all relationship "
        "curves update immediately from the same analytical model.",
        "Q-COMM / experiment workspace",
    )

    metrics_row(ctx)
    st.write("")

    results_tab, journey_tab, relations_tab, export_tab = st.tabs(
        ["Live results", "Photon journey", "Relationships", "Export"]
    )

    # -- live results ------------------------------------------------------
    with results_tab:
        chart_col, text_col = st.columns([1.25, 1.0], gap="large")
        with chart_col:
            st.plotly_chart(fig_link_budget(ctx), config={"displayModeBar": False}, **CHART_FIT)
            st.markdown(
                '<div class="qc-hint">Losses add in decibels: fibre first, then detector. '
                "The final bar is what reaches the detector, in dB relative to launch.</div>",
                unsafe_allow_html=True,
            )
        with text_col:
            st.markdown("### Reading this result")
            st.markdown(
                f"""
A photon launched here has a **{plain(ctx.detection_probability)}** chance of being
registered. That single number is the product of two independent probabilities:

* the fibre passes **{percent(ctx.transmission, 3)}** of photons over {ctx.distance_km:g} km
  ({ctx.attenuation_db_km:g} dB/km, {ctx.fibre_loss_db:.1f} dB total)
* the detector registers **{ctx.detector_efficiency:.0%}** of those that arrive

Expressed in decibels the same statement reads
{ctx.fibre_loss_db:.2f} dB + {ctx.detector_loss_db:.2f} dB = {ctx.total_loss_db:.2f} dB,
and {ctx.dominating_stage} loss dominates this configuration.
"""
            )
            with st.expander("Quantities, units and definitions"):
                definitions = pd.DataFrame(
                    [
                        ("Fibre length L", f"{ctx.distance_km:g}", "km", "Physical span of fibre"),
                        ("Attenuation alpha", f"{ctx.attenuation_db_km:g}", "dB/km", "Loss per kilometre"),
                        ("Fibre loss alphaL", f"{ctx.fibre_loss_db:.3f}", "dB", "Accumulated optical loss"),
                        ("Transmission T", f"{ctx.transmission:.6g}", "-", "Fraction of photons surviving"),
                        ("Detector efficiency eta", f"{ctx.detector_efficiency:.3f}", "-", "Fraction of arrivals registered"),
                        ("Detection probability", f"{ctx.detection_probability:.6g}", "-", "T x eta"),
                        ("Expected arrivals", f"{ctx.expected_arrivals:.6g}", "photons", "N_sent x T"),
                        ("Expected detections", f"{ctx.expected_detections:.6g}", "photons", "N_sent x T x eta"),
                    ],
                    columns=["Quantity", "Value", "Unit", "Definition"],
                )
                st.dataframe(definitions, hide_index=True, **TABLE_FIT)

    # -- photon journey ----------------------------------------------------
    with journey_tab:
        st.plotly_chart(fig_sankey(ctx), config={"displayModeBar": False}, **CHART_FIT)

        breakdown = pd.DataFrame(
            [
                ("Launched", sci(ctx.photons_sent), "100% of the input"),
                ("Lost in fibre", sci(ctx.photons_sent - ctx.expected_arrivals),
                 f"{ctx.fibre_loss_db:.2f} dB of attenuation"),
                ("Arrived at detector", sci(ctx.expected_arrivals),
                 f"{percent(ctx.transmission, 3)} of the input"),
                ("Not registered", sci(ctx.expected_arrivals - ctx.expected_detections),
                 f"{1 - ctx.detector_efficiency:.0%} missed by the detector"),
                ("Detected", sci(ctx.expected_detections),
                 f"{percent(ctx.detection_probability, 3)} of the input"),
            ],
            columns=["Stage", "Photons", "Share and origin"],
        )
        st.dataframe(breakdown, hide_index=True, **TABLE_FIT)
        st.markdown(
            f'<div class="qc-hint">Band widths are proportional to expected counts, so the thin '
            f"thread from the fibre is the honest picture of {percent(ctx.transmission, 2)} survival. "
            "A real run would scatter around these values.</div>",
            unsafe_allow_html=True,
        )

    # -- relationships -----------------------------------------------------
    with relations_tab:
        st.plotly_chart(fig_vs_distance(ctx), config={"displayModeBar": False}, **CHART_FIT)
        st.markdown(
            f'<div class="qc-hint">On the logarithmic axis the exponential decay becomes a straight line '
            f"with slope -alpha/10: one decade of photon number every {ctx.span_10db_km:.0f} km at "
            f"{ctx.attenuation_db_km:g} dB/km.</div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(fig_vs_attenuation(ctx), config={"displayModeBar": False}, **CHART_FIT)
        with right:
            st.plotly_chart(fig_vs_efficiency(ctx), config={"displayModeBar": False}, **CHART_FIT)
        st.markdown(
            '<div class="qc-hint">Each curve varies one parameter and holds the other two fixed, so it '
            "answers a single question: what happens if only this changes.</div>",
            unsafe_allow_html=True,
        )

    # -- export ------------------------------------------------------------
    with export_tab:
        current = pd.DataFrame(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "distance_km": ctx.distance_km,
                    "attenuation_db_per_km": ctx.attenuation_db_km,
                    "detector_efficiency": ctx.detector_efficiency,
                    "photons_sent": ctx.photons_sent,
                    "fibre_loss_db": ctx.fibre_loss_db,
                    "transmission_probability": ctx.transmission,
                    "detection_probability": ctx.detection_probability,
                    "expected_arrivals": ctx.expected_arrivals,
                    "expected_detections": ctx.expected_detections,
                    "photon_loss_percent": ctx.photon_loss_percent,
                }
            ]
        )

        sweep_distances = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        sweep_rows = []
        for distance in sweep_distances:
            alt = SimulationContext.build(
                distance, ctx.attenuation_db_km, ctx.detector_efficiency, ctx.photons_sent
            )
            sweep_rows.append(
                {
                    "distance_km": distance,
                    "fibre_loss_db": alt.fibre_loss_db,
                    "transmission_probability": alt.transmission,
                    "detection_probability": alt.detection_probability,
                    "expected_detections": alt.expected_detections,
                }
            )
        sweep = pd.DataFrame(sweep_rows)

        current_display = pd.DataFrame(
            {
                "Quantity": current.columns,
                "Value": [f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in current.iloc[0]],
            }
        )

        sweep_display = sweep.copy()
        for column in sweep_display.columns:
            sweep_display[column] = [
                f"{v:,.0f}" if column == "distance_km" else f"{v:.4g}" for v in sweep_display[column]
            ]
        sweep_display.columns = ["km", "fibre dB", "T", "P_det", "N_det"]

        left, right = st.columns([1.0, 1.0], gap="large")

        with left:
            st.markdown("### This run")
            st.dataframe(current_display, hide_index=True, **TABLE_FIT)
            st.download_button(
                "Download this run (CSV)",
                current.to_csv(index=False).encode(),
                file_name="qcomm_run.csv",
                mime="text/csv",
                key="dl_run",
                **BTN_FIT,
            )

        with right:
            st.markdown("### Distance sweep")
            st.dataframe(sweep_display, hide_index=True, **TABLE_FIT)
            st.download_button(
                "Download sweep (CSV)",
                sweep.to_csv(index=False).encode(),
                file_name="qcomm_distance_sweep.csv",
                mime="text/csv",
                key="dl_sweep",
                **BTN_FIT,
            )
            st.markdown(
                f'<div class="qc-hint">Same attenuation ({ctx.attenuation_db_km:g} dB/km), detector '
                f"({ctx.detector_efficiency:.0%}) and photon budget; only distance varies.</div>",
                unsafe_allow_html=True,
            )


def page_physics(ctx: SimulationContext) -> None:
    page_head(
        "Physics",
        "The complete model: three equations, the decibel bookkeeping behind them, and the "
        "assumptions that bound what the simulator can claim.",
        "Q-COMM / model documentation",
    )

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown("### 1. Fibre transmission")
        st.latex(r"T = 10^{-\alpha L / 10}")
        st.markdown(
            "`T` is the probability that one photon survives the span, `alpha` the attenuation in "
            "dB/km and `L` the distance in km. The exponential is the whole story of distance: "
            "loss is additive in decibels and multiplicative in photons."
        )

        st.markdown("### 2. Detection probability")
        st.latex(r"P_{\mathrm{det}} = T \times \eta")
        st.markdown(
            "`eta` is detector efficiency as a fraction. A photon must survive the fibre before it "
            "can be registered, so the two stages multiply. Detector inefficiency is not fibre loss."
        )

        st.markdown("### 3. Expected detections")
        st.latex(r"N_{\mathrm{det}} = N_{\mathrm{sent}} \times P_{\mathrm{det}}")
        st.markdown(
            "Scaling the probability by the number of launched photons gives the expected count. "
            "This is an expectation: a real experiment scatters around it."
        )

    with right:
        st.markdown("### Worked example with your settings")
        st.markdown(
            key_value_rows(
                [
                    ("Attenuation, length", f"{ctx.attenuation_db_km:g} dB/km, {ctx.distance_km:g} km"),
                    ("Fibre loss alpha x L", f"{ctx.fibre_loss_db:.2f} dB"),
                    ("Transmission T", plain(ctx.transmission)),
                    ("eta", f"{ctx.detector_efficiency:.2f}"),
                    ("Detection P = T x eta", plain(ctx.detection_probability)),
                    ("N_sent", count_short(ctx.photons_sent)),
                    ("N_det", count_short(ctx.expected_detections)),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        st.latex(
            r"T = 10^{-"
            + f"{ctx.attenuation_db_km:g}"
            + r"\times "
            + f"{ctx.distance_km:g}"
            + r"/10} = "
            + latex_number(ctx.transmission)
        )
        st.markdown(
            '<div class="qc-sub">Substituting the current parameters into the same expressions the '
            "simulator evaluates. No lookup tables are involved at any point.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)

    with st.expander("Why decibels, and what the numbers mean"):
        st.markdown(
            f"""
Attenuation is quoted in decibels because each metre of fibre multiplies the surviving
amplitude independently; logarithms turn that chain of multiplications into a sum.

| Loss | Photon fraction | Meaning |
| --- | --- | --- |
| 3 dB | 0.50 | half the photons |
| 10 dB | 0.10 | one decade |
| 20 dB | 0.01 | one photon in a hundred |

At {ctx.attenuation_db_km:g} dB/km that means 3 dB every {ctx.span_3db_km:.0f} km and 10 dB every
{ctx.span_10db_km:.0f} km. Conversions: `L_dB = -10 log10(T)` and `T = 10^(-L_dB/10)`.
"""
        )

    with st.expander("Reference values (context only - not model inputs)"):
        st.markdown(
            """
| Component | Typical figure | Note |
| --- | --- | --- |
| Standard single-mode fibre, 1550 nm | ~0.20 dB/km | Telecommunications C-band |
| Ultra-low-loss fibre, 1550 nm | 0.16-0.18 dB/km | Premium long-haul spans |
| Standard single-mode fibre, 1310 nm | ~0.35 dB/km | Zero-dispersion window |
| InGaAs SPAD, 1550 nm | 10-30% efficiency | Room temperature to modest cooling |
| Superconducting nanowire detector | above 90% efficiency | Cryogenic, low dark counts |
| Connector or splice | 0.1-0.5 dB each | Not represented by this model |

These are indicative figures for orientation. The simulator only ever uses the values you set
in the sidebar.
"""
        )

    st.markdown("### Assumptions and limitations")
    st.markdown(
        f"""
**Included.** Uniform attenuation `alpha` over the entire span; a single scalar detector
efficiency `eta`; expectation values computed analytically for any number of launched photons.

**Not included.** Dark counts, afterpulsing, dead time, timing jitter and background light;
connectors, splices and bend loss; amplifiers, repeaters, trusted nodes and quantum memories;
polarisation drift, dispersion and nonlinear effects; atmospheric, free-space and satellite
channels; multi-photon emission, decoy states, error correction and privacy amplification.

**Where it would break first.** The absence of a dark-count floor. This configuration predicts
{count_short(ctx.expected_detections)} detections from {count_short(ctx.photons_sent)} photons,
while a real detector contributes dark counts every second regardless of signal. Once the signal
rate falls below that floor, the model's {plain(ctx.detection_probability)} stops being
measurable even though the arithmetic remains correct.
"""
    )
    st.markdown(
        '<div class="qc-panel"><span class="qc-key-label">This is not a QKD implementation. '
        "Q-COMM models the transmission and detection layer only: no basis reconciliation, no key "
        "rate, no security analysis, and no experimental data.</span></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Assistant surfaces
# ----------------------------------------------------------------------------


def _answer(question: str) -> None:
    ctx = st.session_state.get("_ctx")
    if ctx is None:
        return
    messages = st.session_state.assistant_messages
    messages.append({"role": "user", "content": question})
    messages.append(
        {
            "role": "assistant",
            "content": local_assistant(question, ctx, st.session_state.get("assistant_level", DEFAULT_LEVEL)),
        }
    )
    del messages[:-40]


def _render_messages(compact: bool = False) -> None:
    messages = st.session_state.assistant_messages
    if not messages:
        if compact:
            text = ("Ask about this run: attenuation, loss, detector efficiency, distance, "
                    "or a what-if. Answers use your live parameters.")
        else:
            text = ("Ask about the current run - attenuation, loss, detector efficiency, "
                    "distance, the diagrams, or a hypothetical such as doubling the span. "
                    "Answers use your live parameters.")
        st.markdown(f'<div class="qc-sub">{text}</div>', unsafe_allow_html=True)
        return

    for index, message in enumerate(messages):
        if message["role"] == "user":
            st.markdown(
                f'<div class="qc-msg-user">{html.escape(message["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.container(key=f"qcmsg-{index}"):
                st.markdown(message["content"])


def _level_control(compact: bool = False) -> None:
    ss = st.session_state
    ss.setdefault("qc_level", ss.get("assistant_level", DEFAULT_LEVEL))

    if compact:
        st.markdown('<div class="qc-side-label">Explanation level</div>', unsafe_allow_html=True)

    level = None
    if hasattr(st, "segmented_control"):
        level = st.segmented_control(
            "Explanation level",
            list(LEVELS),
            key="qc_level",
            label_visibility="collapsed" if compact else "visible",
            help="Applied to new answers. Earlier replies are not rewritten.",
        )

    if level is None:
        level = st.selectbox(
            "Explanation level",
            list(LEVELS),
            index=list(LEVELS).index(ss.get("assistant_level", DEFAULT_LEVEL)),
            key="qc_level_select",
            label_visibility="collapsed" if compact else "visible",
            help="Applied to new answers. Earlier replies are not rewritten.",
        )

    ss["assistant_level"] = level

def _chat_input(form_key: str) -> None:
    with st.form(key=form_key, clear_on_submit=True, border=False):
        text_col, send_col = st.columns([3.9, 1.1] if "panel" in form_key else [6.0, 1.0], gap="small")
        with text_col:
            text = st.text_input(
                "Message",
                placeholder="Ask about this run...",
                label_visibility="collapsed",
            )
        with send_col:
            sent = st.form_submit_button("Send", **BTN_FIT)

    if sent and text and text.strip():
        _answer(text.strip())
        st.rerun()


def page_assistant(ctx: SimulationContext) -> None:
    page_head(
        "Research assistant",
        "A local, rule-based assistant that reads your current simulation and explains what it "
        "means. No API key, no external model, no network access.",
        "Q-COMM / interpretation",
    )

    head_col, clear_col = st.columns([3.0, 1.0], gap="medium", vertical_alignment="bottom")
    with head_col:
        _level_control(compact=True)
    with clear_col:
        if st.button("Clear conversation", key="clear_full", help="Remove every message"):
            st.session_state.assistant_messages = []
            st.rerun()

    st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)

    if st.session_state.assistant_messages:
        with st.container(height=520, border=True, autoscroll=True):
            _render_messages()
    else:
        with st.container(border=True):
            _render_messages()
            st.markdown("")
            columns = st.columns(2, gap="small")
            for index, question in enumerate(SUGGESTED_QUESTIONS[:6]):
                if columns[index % 2].button(question, key=f"sug_full_{index}", **BTN_FIT):
                    _answer(question)
                    st.rerun()

    _chat_input("qc_form_page")

    st.markdown(
        f'<div class="qc-hint">{html.escape(ASSISTANT_NOTE)}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("What the assistant can answer"):
        topics = pd.DataFrame(HELP_TOPICS, columns=["Topic", "Example question"])
        st.dataframe(topics, hide_index=True, **TABLE_FIT)
        st.markdown(
            "It matches keywords against a fixed topic index, then composes the answer from that "
            "topic plus the live simulation values. If nothing matches it says so instead of "
            "guessing."
        )

def _close_panel() -> None:
    st.session_state.assistant_open = False
    st.session_state.assistant_min = False


def render_floating_assistant() -> None:
    ss = st.session_state

    if ss.assistant_open and ss.assistant_min:
        with st.container(key="qc_mini"):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center", gap="small"):
                st.markdown(
                    '<div class="qc-panel-title">Q-COMM assistant</div>', unsafe_allow_html=True
                )
                if st.button("Open", key="qc_restore", help="Restore panel"):
                    ss.assistant_min = False
                    st.rerun()
                if st.button("×", key="qc_close_mini", help="Close"):
                    _close_panel()
                    st.rerun()
        return

    if ss.assistant_open:
        if ss.assistant_max:
            st.markdown(
                "<style>.st-key-qc_panel{width:min(580px,calc(100vw - 40px));}</style>",
                unsafe_allow_html=True,
            )

        with st.container(key="qc_panel"):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center", gap="small"):
                st.markdown(
                    '<div class="qc-panel-title">Q-COMM assistant</div>'
                    '<div class="qc-panel-sub">Rule-based / offline</div>',
                    unsafe_allow_html=True,
                )
                if st.button("–", key="qc_min", help="Minimise to a strip"):
                    ss.assistant_min = True
                    st.rerun()
                if st.button("Max", key="qc_max", type="primary" if ss.assistant_max else "secondary",
                             help="Restore size" if ss.assistant_max else "Maximise"):
                    ss.assistant_max = not ss.assistant_max
                    st.rerun()
                if st.button("×", key="qc_close", help="Close"):
                    _close_panel()
                    st.rerun()

            _level_control(compact=True)

            if ss.assistant_messages:
                height = 500 if ss.assistant_max else 292
                with st.container(height=height, border=False, autoscroll=True):
                    _render_messages(compact=True)
            else:
                _render_messages(compact=True)
                chips = st.columns(2, gap="small")
                quick = (
                    "Why is my result so low?",
                    "What if I double the distance?",
                    "Explain the diagram",
                    "How can I improve it?",
                )
                for index, question in enumerate(quick):
                    if chips[index % 2].button(question, key=f"chip_{index}", **BTN_FIT):
                        _answer(question)
                        st.rerun()

            _chat_input("qc_form_panel")

            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                if st.button("Clear", key="qc_clear", help="Clear conversation"):
                    ss.assistant_messages = []
                    st.rerun()
                st.markdown(
                    '<div class="qc-hint">Local rules, no model.</div>', unsafe_allow_html=True
                )
        return

    with st.container(key="qc_launcher"):
        if st.button("◎  Assistant", key="qc_launch", help="Open the Q-COMM assistant"):
            ss.assistant_open = True
            ss.assistant_min = False
            st.rerun()

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="qc-brand">Q<span>-</span>COMM</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qc-brand-sub">Photon transmission simulator</div>', unsafe_allow_html=True
        )
        st.markdown("")

        if "_goto" in st.session_state:
            st.session_state["qc_nav"] = st.session_state.pop("_goto")
        st.session_state.setdefault("qc_nav", PAGES[0])
        page = st.radio("Section", PAGES, key="qc_nav", label_visibility="collapsed")

        st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="qc-side-label">Experiment parameters</div>', unsafe_allow_html=True)

        linked_param(
            "distance", "Fibre length (km)", "km", integer=True,
            help_text="Physical span of fibre between source and detector.",
        )
        linked_param(
            "attenuation", "Attenuation (dB/km)", "dB/km",
            help_text="Optical loss per kilometre. 0.20 dB/km is standard single-mode fibre at 1550 nm.",
        )
        linked_param(
            "efficiency", "Detector efficiency (%)", "%", integer=True,
            help_text="Share of arriving photons the detector successfully registers.",
        )
        linked_param(
            "log_photons", "Photons launched (10^x)", "10^x",
            help_text="Input photon budget, given as a power of ten.",
        )

        photons = int(round(10 ** st.session_state["p_log_photons"]))
        st.caption(f"Launching {photons:,} photons")

        st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)

        with st.expander("Reference hardware"):
            st.caption("Indicative values. Selecting an option applies it to the controls above.")
            st.selectbox(
                "Fibre", list(FIBRE_PRESETS), key="fibre_preset", on_change=_apply_fibre_preset
            )
            st.selectbox(
                "Detector", list(DETECTOR_PRESETS), key="detector_preset", on_change=_apply_detector_preset
            )

        if st.button("Reset to defaults", key="reset_params", **BTN_FIT):
            st.session_state["_do_reset"] = True
            st.rerun()

        st.markdown(
            '<div class="qc-hint">Analytical model, single-mode silica fibre. '
            "Outputs are theoretical expectations, not measurements.</div>",
            unsafe_allow_html=True,
        )

    return page


def main() -> None:
    init_state()
    apply_pending_reset()

    page = render_sidebar()

    params = {
        "distance": st.session_state["p_distance"],
        "attenuation": st.session_state["p_attenuation"],
        "efficiency": st.session_state["p_efficiency"] / 100.0,
        "photons": int(round(10 ** st.session_state["p_log_photons"])),
    }

    ctx = SimulationContext.build(
        distance_km=params["distance"],
        attenuation_db_km=params["attenuation"],
        detector_efficiency=params["efficiency"],
        photons_sent=params["photons"],
    )
    st.session_state["_ctx"] = ctx

    if page == "Overview":
        page_overview(ctx)
    elif page == "Simulator":
        page_simulator(ctx)
    elif page == "Physics":
        page_physics(ctx)
    else:
        page_assistant(ctx)

    if page != "Research Assistant":
        render_floating_assistant()

    st.markdown('<div class="qc-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qc-foot">Q-COMM · analytical photon-transmission model · '
        "every figure is a theoretical expectation under the assumptions listed in Physics.</div>",
        unsafe_allow_html=True,
    )


main()
