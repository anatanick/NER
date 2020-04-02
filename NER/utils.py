import os
import logging
import subprocess
import re

def conlleval(prediction):
    prediction_str = list()
    for sent in prediction:
        for char, tag, ptag in sent:
            prediction_str.append("{} {} {}\n".format(char, tag, ptag))
        prediction_str.append("\n")
    prediction_str = "".join(prediction_str)

    proc = subprocess.Popen(
        ["perl", "./conlleval.pl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_value, stderr_value = proc.communicate(prediction_str.encode("utf-8"))
    evaluate_str = stdout_value.decode("utf-8")
    logging.info(evaluate_str)

    pattern_str = "\s*(\w+):.* precision:\s+([\d\.]+)%; recall:\s+([\d\.]+)%;\s+FB1:\s+([\d\.]+).*"
    pattern = re.compile(pattern_str)

    result = dict()
    for line in evaluate_str.split('\n'):
        m = pattern.match(line)
        if m is None:
            continue
        result[m.group(1)] = {
            "precision": float(m.group(2)) / 100,
            "recall": float(m.group(3)) / 100,
            "FB1": float(m.group(4)) / 100,
        }
    return result
