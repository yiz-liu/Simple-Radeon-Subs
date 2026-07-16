def parse_out_time_seconds(line: str) -> float | None:
    key, separator, raw_value = line.partition("=")
    if separator != "=" or key != "out_time_us":
        return None

    try:
        microseconds = int(raw_value)
    except ValueError:
        return None

    if microseconds < 0:
        return None
    return microseconds / 1_000_000
