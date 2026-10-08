#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查找 / 结束残留的训练进程（只在 Linux 上有效；run_6gpu.sh 启动前调用）。

残留进程指：
  - 当前用户的训练进程：命令行里有 run.py train / torch.distributed.run /
    streetlearn.python.ppo_train 的进程（含包装脚本），以及它们的全部子进程
    （环境 worker、resource tracker）；
  - 锁着数据集副本缓存目录（--cache_dir）里 LevelDB LOCK 的进程：之前的训练
    崩掉或卡住后没有退出的环境 worker。它们让新 run 打不开副本（IO error: lock
    .../LOCK: Resource temporarily unavailable），还占着内存。

用法：
  python -m streetlearn.tools.stale_procs --cache_dir tmp_ds          # 列出；有残留时退出码 1
  python -m streetlearn.tools.stale_procs --cache_dir tmp_ds --kill   # 结束（先 TERM，等一会再 KILL）
"""
from __future__ import print_function

import argparse
import glob
import os
import re
import signal
import sys
import time

MAIN_RE = re.compile(
    r"run\.py train|torch\.distributed\.run|streetlearn\.python\.ppo_train")
OUTPUT_DIR_RE = re.compile(r"--output_dir[= ](\S+)")


def _read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def _cmdline(pid):
    try:
        raw = _read("/proc/%d/cmdline" % pid, "rb")
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def _stat(pid):
    """(state, ppid, start_ticks)；进程已不存在时返回 None。"""
    try:
        raw = _read("/proc/%d/stat" % pid)
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2:].split()   # 从第 3 个字段（state）开始
    return fields[0], int(fields[1]), int(fields[19])


def _uid(pid):
    try:
        return os.stat("/proc/%d" % pid).st_uid
    except OSError:
        return None


def _ancestors():
    """本进程及其祖先（不能把调用方自己算成残留）。"""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        st = _stat(pid)
        if st is None:
            break
        pid = st[1]
    return pids


def _lock_holders(cache_dir):
    """持有 cache_dir/streetlearn_ds_*/LOCK 文件锁的进程（按设备号 + inode 匹配 /proc/locks）。"""
    files = set()
    for path in glob.glob(os.path.join(cache_dir, "streetlearn_ds_*", "LOCK")):
        try:
            st = os.stat(path)
        except OSError:
            continue
        files.add((os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))
    holders = set()
    if not files:
        return holders
    try:
        lines = _read("/proc/locks").splitlines()
    except OSError:
        return holders
    for line in lines:
        parts = line.split()
        if "->" in parts:          # 等锁的请求，不是持有者
            continue
        for i, token in enumerate(parts):
            pieces = token.split(":")
            if len(pieces) == 3:   # MAJOR:MINOR:INODE（主次设备号是十六进制）
                try:
                    key = (int(pieces[0], 16), int(pieces[1], 16), int(pieces[2]))
                except ValueError:
                    break
                if key in files and parts[i - 1].isdigit():
                    holders.add(int(parts[i - 1]))
                break
    return holders


def find_stale(cache_dir):
    """返回 (训练主进程集合, 全部残留进程集合, {pid: (state, ppid, start_ticks)})。"""
    skip = _ancestors()
    uid = os.getuid()
    procs = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = _stat(int(name))
            if st is not None and st[0] != "Z":
                procs[int(name)] = st
    main = {pid for pid in procs
            if pid not in skip and _uid(pid) == uid and MAIN_RE.search(_cmdline(pid))}
    children = {}
    for pid, (_, ppid, _) in procs.items():
        children.setdefault(ppid, []).append(pid)
    stale, stack = set(main), list(main)
    while stack:   # 训练进程的全部子孙：环境 worker、resource tracker……
        for child in children.get(stack.pop(), []):
            if child not in stale and child not in skip:
                stale.add(child)
                stack.append(child)
    stale |= {pid for pid in _lock_holders(cache_dir) if pid in procs and pid not in skip}
    return main, stale, procs


def _age(seconds):
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    if days:
        return "%dd %02dh" % (days, hours)
    if hours:
        return "%dh %02dm" % (hours, rem // 60)
    return "%dm" % (rem // 60)


def _newest_mtime(path):
    newest = None
    for root, _, files in os.walk(path):
        for name in files:
            try:
                mtime = os.path.getmtime(os.path.join(root, name))
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
    return newest


def describe(main, stale, procs):
    now = time.time()
    boot = 0
    for line in _read("/proc/stat").splitlines():
        if line.startswith("btime "):
            boot = int(line.split()[1])
    hz = os.sysconf("SC_CLK_TCK")

    def started(pid):
        return boot + procs[pid][2] / float(hz)

    oldest = min(started(pid) for pid in stale)
    print("[stale] %d leftover training process(es) are still alive; the oldest "
          "started %s (%s ago)." % (len(stale), time.strftime(
              "%Y-%m-%d %H:%M", time.localtime(oldest)), _age(now - oldest)))
    print("残留的训练进程：共 %d 个（%d 个训练 / 启动进程，%d 个环境 worker 等子进程）。"
          % (len(stale), len(main), len(stale - main)))
    shown = sorted(main) or sorted(stale)[:10]
    print("  %7s %7s  %-11s  %-8s  %s" % ("PID", "PPID", "STARTED", "AGE", "COMMAND"))
    for pid in shown:
        cmd = _cmdline(pid)
        print("  %7d %7d  %-11s  %-8s  %s" % (
            pid, procs[pid][1], time.strftime("%m-%d %H:%M", time.localtime(started(pid))),
            _age(now - started(pid)), cmd if len(cmd) <= 110 else cmd[:107] + "..."))
    if len(shown) < len(stale):
        print("  ……另有 %d 个子进程未列出（环境 worker：spawn_main / resource_tracker）"
              % (len(stale) - len(shown)))
    # 这些进程在写哪个目录、最后一次写是什么时候：判断那次训练是还在跑还是早就卡住了。
    dirs = {}
    for pid in main:
        match = OUTPUT_DIR_RE.search(_cmdline(pid))
        if not match:
            continue
        path = match.group(1)
        if not os.path.isabs(path):
            try:
                path = os.path.join(os.readlink("/proc/%d/cwd" % pid), path)
            except OSError:
                continue
        dirs[os.path.normpath(path)] = None
    for path in sorted(dirs):
        newest = _newest_mtime(path) if os.path.isdir(path) else None
        print("  它们的输出目录 %s：%s" % (
            path, "最后写入 %s（%s 前）" % (
                time.strftime("%m-%d %H:%M", time.localtime(newest)), _age(now - newest))
            if newest else "没有文件"))


def _alive(pids):
    return {pid for pid in pids
            if (_stat(pid) or ("Z",))[0] != "Z"}


def kill(stale, grace):
    """先 SIGTERM，最多等 grace 秒；还活着的再 SIGKILL。返回仍然活着的进程。"""
    denied = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        targets = sorted(_alive(stale), reverse=True)
        if not targets:
            break
        print("[stale] %s -> %d process(es)" % (
            "SIGTERM" if sig == signal.SIGTERM else "SIGKILL", len(targets)))
        for pid in targets:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                denied.append(pid)
        deadline = time.time() + (grace if sig == signal.SIGTERM else 5.0)
        while time.time() < deadline and _alive(stale):
            time.sleep(0.5)
    if denied:
        print("[stale] 没有权限结束：%s（属于别的用户？）" % " ".join(map(str, sorted(set(denied)))))
    return _alive(stale)


def main():
    p = argparse.ArgumentParser(description="查找 / 结束残留的训练进程（Linux）")
    p.add_argument("--cache_dir", required=True, help="数据集副本缓存目录（ds_cache_dir）")
    p.add_argument("--kill", action="store_true", help="结束找到的进程")
    p.add_argument("--grace", type=float, default=20.0,
                   help="SIGTERM 之后最多等几秒再 SIGKILL（默认 20）")
    p.add_argument("--pid_file", default=None,
                   help="把找到的 PID（空格分隔）写进这个文件，便于手动 kill $(cat 文件)")
    args = p.parse_args()

    if not os.path.isdir("/proc/self"):
        return 0          # 非 Linux：没有 /proc，无从检查
    main_pids, stale, procs = find_stale(args.cache_dir)
    if args.pid_file:
        with open(args.pid_file, "w") as f:
            f.write(" ".join(str(pid) for pid in sorted(stale)) + "\n")
    if not stale:
        if args.kill:
            print("[stale] 没有残留的训练进程。")
        return 0
    describe(main_pids, stale, procs)
    if not args.kill:
        return 1
    left = kill(stale, args.grace)
    _, still, _ = find_stale(args.cache_dir)
    still |= left
    if still:
        print("[stale] 仍有 %d 个进程没有退出：%s" % (
            len(still), " ".join(map(str, sorted(still)))))
        return 1
    print("[stale] 已全部结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
