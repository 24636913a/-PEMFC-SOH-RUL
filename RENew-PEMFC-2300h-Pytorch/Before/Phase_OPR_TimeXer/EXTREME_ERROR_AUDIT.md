# Extreme error audit

The top 20 P2 errors were selected from Fold 3 at 12 h after deduplicating true target time. They represent 20 prediction points and 20 unique target timestamps. The largest is 593.411977 mV at target age 1462.25 h, with P2 correction -567.174744 mV.

These unique targets cluster around late-development operating ages rather than representing 20 independent events. Several have large predicted-current or q error, but others have small input-current mismatch and still large correction. Therefore future-condition error is a contributing observation, not a complete explanation. No point was removed or corrected. The top-20 squared-error share within Fold 3 / 12 h is 0.868; it does not justify calling the entire pattern a single-outlier artifact.
