import os
import sys

shell_str = "python main.py"
device = input("input device:")
additional_arg = input("input additional args:")

for i in range(1, len(sys.argv)):
    print('参数 %s 为：%s' % (i, sys.argv[i]))
    shell_str += " " + str(sys.argv[i])

shell_str += " --work-dir ./work_dir/uav/jm_" + device + " --device " + device
shell_str += " " + additional_arg

result = os.system(shell_str)
while result != 0:
    print(result)
    print("train fail, restart")
    print(shell_str)
    result = os.system(shell_str)
