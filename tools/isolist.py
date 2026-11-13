# lists directories of an Xbox disc image (XDVDFS): isolist.py <iso> <path-filter-substring>
import struct, sys

MAGIC = b"MICROSOFT*XBOX*MEDIA"
PARTITIONS = (0, 0x18300000, 0xFD90000, 0x2080000)


def find_partition(f):
    for part in PARTITIONS:
        f.seek(part + 32 * 2048)
        if f.read(20) == MAGIC:
            return part
    return None


def read_table(f, base, sector, size):
    f.seek(base + sector * 2048)
    return f.read(size)


def entries(table):
    # walk the b-tree of one directory table in order
    out = []
    stack = [0]
    seen = set()
    while stack:
        off = stack.pop()
        if off in seen or off * 4 + 14 > len(table):
            continue
        seen.add(off)
        left, right, sector, size, attr, nlen = struct.unpack_from("<HHIIBB", table, off * 4)
        name = table[off * 4 + 14: off * 4 + 14 + nlen].decode("latin-1")
        if left != 0xFFFF:
            out.append((name, sector, size, attr))
        for nxt in (right, left):
            if nxt and nxt != 0 and nxt * 4 < len(table):
                stack.append(nxt)
    return out


def main():
    iso = sys.argv[1]
    want = sys.argv[2].lower() if len(sys.argv) > 2 else ""
    with open(iso, "rb") as f:
        base = find_partition(f)
        if base is None:
            sys.exit("not an Xbox disc image")
        f.seek(base + 32 * 2048 + 20)
        root_sector, root_size = struct.unpack("<II", f.read(8))
        todo = [("", root_sector, root_size)]
        while todo:
            path, sector, size = todo.pop()
            table = read_table(f, base, sector, size)
            for name, sec, sz, attr in entries(table):
                full = path + "/" + name if path else name
                isdir = bool(attr & 0x10)
                if want in full.lower():
                    print(("D " if isdir else "F ") + full + ("" if isdir else " " + str(sz)))
                if isdir:
                    todo.append((full, sec, sz))


main()
