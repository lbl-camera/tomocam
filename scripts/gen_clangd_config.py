import re
import json
from pathlib import Path

# valid extensions
exts = ('.cpp', '.cu')

def find_source_files():

    sources = set()
    with open('CMakeLists.txt') as f:
        lines = f.readlines()

    pat = re.compile(r'set\(SRC')
    for i, line in enumerate(lines):
        if re.search(pat, line):
            break

    # serch lines for files
    jbegin = i + 1
    while True:
        srcpth = Path(lines[jbegin].strip())
        srcfile = Path(*srcpth.parts[1:])
        if srcfile.suffix in exts and srcfile.exists():
            sources.add(srcfile)
            jbegin += 1
            if jbegin > len(lines):
                break
        else:
            break
    return sources

def gen_compile_cmd(pth, root_dir):

    if isinstance(root_dir, Path):
        root_dir = root_dir.as_posix()
    if pth.suffix == '.cu':
        cmd = f"clang++ -x cuda -D__NVCC__ -std=c++20 -I./src -c {pth.as_posix()}"
    else:
        cmd = f"clang++ -std=c++20 -I./src -c {pth.as_posix()}"
    return { "directory": root_dir, "command": cmd, "file": pth.as_posix() }

def main():

    sources =find_source_files()
    root_dir = Path.cwd()

    compile_commands = []
    for srcf in sources:
        # make path absoule
        src_abs_path = srcf.absolute()
        entry = gen_compile_cmd(src_abs_path, root_dir)
        compile_commands.append(entry)

    """ Debug
    for cmd in compile_commands:
        print(cmd)
        break
    """
    with open('compile_commands.json', 'w') as f:
        json.dump(compile_commands, f, indent=4)
        
if __name__ == '__main__':
    main()
