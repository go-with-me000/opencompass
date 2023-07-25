import fnmatch
import os
import re
import subprocess

basepath = '.'

cmd = 'git ls-files'
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=basepath)
p.wait()
lines = p.stdout.readlines()
filenames = [os.path.join(basepath, line.decode('utf-8').strip()) for line in lines]

with open(".github_blacklist", "r") as f:
    internal_patterns = f.readlines()
internal_patterns = [os.path.join(basepath, pattern.strip()) for pattern in internal_patterns]

for filename in filenames:
    if any(fnmatch.fnmatch(filename, ip) for ip in internal_patterns):
        # delete the file
        print("deleting file: {}".format(filename))
        os.remove(filename)
        continue

    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            if re.match(r'^.*# COPYBARA_INTERNAL$', line.strip()):
                continue
            elif re.match(r'^.*# COPYBARA_EXTERNAL.*$', line.strip()):
                line = line.replace("# COPYBARA_EXTERNAL ", "")
            f.write(line)
