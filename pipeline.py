import subprocess
import argparse
import re
import tempfile
from config import *
from utils import *

pattern = re.compile(r"Results saved to (.+)")

def pipeline(args):
    # resp generate first
    model = args.model
    resp_gen_arg_list = ["python", "resp_gen.py"]
    for key, value in vars(args).items():
        resp_gen_arg_list.append("--" + key)
        resp_gen_arg_list.append(str(value))
    print(' '.join(resp_gen_arg_list))

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        print(f"Monitor Output at :  tail -f {temp_file.name}")
        result = subprocess.run(resp_gen_arg_list, stdout=temp_file, stderr=subprocess.STDOUT)

        temp_file.seek(0)
        stdout = temp_file.read()

        match = pattern.search(stdout)
        if match:
            filepath = match.group(1)
            print(filepath)
        else:
            print(result.stderr)
            raise ValueError("No file path found in the output.")
        subprocess.run(["python", "resp_eval.py", "--file_path", filepath])


if __name__ == '__main__':
    args = get_args()
    print(args)
    pipeline(args)