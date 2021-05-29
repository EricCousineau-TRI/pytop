""" pytop.py: Htop copycat implemented in Python. """

__author__ = 'Andrii Oshtuk, Eric Cousinea'
__copyright__ = '(C) 2021 ' + __author__
__license__ = "MIT"
__version__ = '1.1.0'

import argparse
from collections import namedtuple
from datetime import timedelta
import os
import re
import shlex
import time

import psutil
import urwid

# Artisinal pieces.

def reformat_command(s):
    s = re.sub(r"/.*?\.runfiles/", "{runfiles}/", s)
    s = re.sub(r"/.*?/bazel-bin/", "{bin}", s)
    return s


def _command_sorting_key(p):
    cmd = p.full_command
    # Put custom commands at top.
    if "/anzu/" in cmd:
        return 0
    return 1


def _sched_sorting_key(sched):
    s = sched_str(sched)
    if s == "rr":
        return (0, s)
    elif s == "fifo":
        return (1, s)
    else:
        return (2, s)


def custom_process_sorting_key(p):
    return (
        _command_sorting_key(p),
        _sched_sorting_key(p.scheduler),
        -p.priority,
        human_sorting_key(p.command),
    )

def custom_process_filter(p):
    cmd = p.full_command
    if "bazel(anzu)" in cmd:
        return False
    if "java" in cmd:
        return False
    if "/anzu/" in cmd:
        return True
    if "ksoftirqd" in cmd:
        return True
    if "nv_queue" in cmd:
        return True
    if "nvidia" in cmd:
        return True
    if affinity_str(p.cpu_affinity) != "":
        return True
    return False


def get_cols():
    return [
        ProcessColumn("USER", "{:<11}", lambda pr: f"{pr.user[:9]}"),
        ProcessColumn("PID", "{:<7}", lambda pr: f"{pr.pid}"),
        ProcessColumn("CPU%", "{:<10}", lambda pr: f"{pr.cpu_percent}"),
        ProcessColumn("NLWP", "{:<6}", lambda pr: f"{pr.num_threads}"),
        ProcessColumn("Sched", "{:<10}", lambda pr: sched_str(pr.scheduler)),
        ProcessColumn("Prio", "{:<6}", lambda pr: str(pr.priority)),
        ProcessColumn("CPU Affinity", "{:<15}", lambda pr: affinity_str(pr.cpu_affinity)),
        ProcessColumn("Command", "{:<60}", lambda pr: reformat_command(pr.command)[:60]),
    ]


# Base structure.


class Process:
    """
    Information about running process with PID.

    Slightly different rendering of what psutil.Process provides.
    """

    def __init__(self, pid):
        assert isinstance(pid, int)
        self.pid = pid

        self.command = ''
        self.user = None
        self.nice = None
        self.cpu_percent = None
        self.status = None
        self.num_threads = None
        self.cpu_affinity = None
        self.scheduler = None
        self.priority = None

        self.ppid = None
        self.parent = None
        self.parent_list = None  # closest parent first

    def update(self, p):
        """Reads information from a psutil.Process object."""
        assert isinstance(p, psutil.Process)
        self.full_command = get_command_or_name(p, short=False)
        self.command = get_command_or_name(p, short=True)
        self.user = p.username()
        self.nice = p.nice()
        self.cpu_percent = p.cpu_percent()
        self.status = p.status()
        self.num_threads = p.num_threads()
        self.cpu_affinity = p.cpu_affinity()
        self.scheduler = os.sched_getscheduler(p.pid)
        self.priority = os.sched_getparam(p.pid).sched_priority

        self.ppid = p.ppid()
        self.parent = None
        self.parent_list = []


class ProcessesController:
    def __init__(self):
        self._process_map = dict()

    def update(self, process_filter, with_threads):
        # Filter can't use parent info.
        actual_pids = set()
        for p, thread_parent in process_iter(with_threads):
            with p.oneshot():
                actual_pids.add(p.pid)
                if p.pid not in self._process_map:
                    process = Process(p.pid)
                else:
                    process = self._process_map[p.pid]
                process.update(p)
            if thread_parent is not None:
                process.ppid = thread_parent.pid
                process.num_threads = -1
            if not process_filter(process):
                continue
            self._process_map[p.pid] = process
        # Remove old.
        obsolete_pids = set(self._process_map.keys()) - actual_pids
        for pid in obsolete_pids:
            del self._process_map[pid]
        # Denote parents.
        for pr in self._process_map.values():
            pr.parent = self._process_map.get(pr.ppid)
        for pr in self._process_map.values():
            node = pr.parent
            while node is not None:
                pr.parent_list.append(node)
                node = node.parent

    @property
    def processes(self):
        return list(self._process_map.values())


ProcessColumn = namedtuple("ProcessColumn", ("name", "fmt", "value"))


class ProcessEntry(urwid.Text):
    # N.B. We use this weird inheritance to associate a process with a widget (Text). This way,
    # we can use urwid's ModifiedList (via SimpleFocusListWalker) to facilitate real-time updates.
    def __init__(self, pr, *args, **kwargs):
        self.pr = pr
        super().__init__(*args, **kwargs)


class ProcessPanel(urwid.WidgetWrap):
    """docstring for ProcessPanel"""
    def __init__(self, controller):
        self.controller = controller

        self.cols = get_cols()
        header_text = ""
        for col in self.cols:
            header_text += col.fmt.format(col.name)
        self.header = urwid.Text(('table_header', header_text))

        self.entries = urwid.SimpleFocusListWalker([])
        self.table_view = urwid.ListBox(self.entries)
        self.table_widget = urwid.Frame(self.table_view, header=self.header)

        self._tree = None
        urwid.WidgetWrap.__init__(self, self.table_widget)

    def _process_sort_key(self, pr):
        key = custom_process_sorting_key(pr)
        if self._tree and pr.parent is not None:
            key = self._process_sort_key(pr.parent) + key
        return key

    def _process_row_text(self, pr):
        result = []
        for col in self.cols:
            text = col.fmt.format(col.value(pr))
            if self._tree and col.name == "Command":
                level = len(pr.parent_list)
                indent = "  " * level
                text = indent + text
            result.append(text)
        return result

    def refresh(self, tree):
        self._tree = tree
        # Register new processes.
        prev_processes = [x.pr for x in self.entries]
        for pr in self.controller.processes:
            if pr not in prev_processes:
                self.entries.append(ProcessEntry(pr, ""))
        # Remove old processes.
        # Copy to admit mutation while iterating.
        for entry in list(self.entries):
            if entry.pr not in self.controller.processes:
                self.entries.remove(entry)
        # Sort processes.
        # TODO(eric.cousineau): Let the index follow the sorting?
        _, focus = self.entries.get_focus()
        self.entries.sort(key=lambda entry: self._process_sort_key(entry.pr))
        self.entries.set_focus(focus)
        # Update entries.
        for entry in self.entries:
            entry.set_text(self._process_row_text(entry.pr))


class Application:
    palette = [
        ('foot', 'black', 'dark cyan'),
        ('normal', 'white', ''),
        ('fields_names', 'dark cyan', ''),
        ('table_header', 'black', 'dark green'),
    ]

    def __init__(
        self,
        *,
        refresh_rate_sec,
    ):
        # options
        self.refresh_rate_sec = refresh_rate_sec

        # data sources
        self.processes = ProcessesController()

        # buttons
        urwid.Button.button_left = urwid.Text("")
        urwid.Button.button_right = urwid.Text("")

        self.w_pause = urwid.Button([('normal', 'F2 '), ('foot', ' Pause ')])
        urwid.connect_signal(self.w_pause, 'click', self.pause)

        self.w_tree = urwid.Button([('normal', 'F3 '), ('foot', ' Flat ')])
        urwid.connect_signal(self.w_tree, 'click', self.tree)

        self.w_thread = urwid.Button([('normal', 'F4 '), ('foot', ' Threads ')])
        urwid.connect_signal(self.w_thread, 'click', self.thread)

        self.w_quit = urwid.Button([('normal', 'F10 '), ('foot', ' Quit ')])
        urwid.connect_signal(self.w_quit, 'click', self.quit)

        # widgets
        self.buttons = urwid.Columns([
            self.w_pause,
            self.w_tree,
            self.w_thread,
            self.w_quit,
        ])
        self.processes_list = ProcessPanel(self.processes)
        self.main_widget = urwid.Frame(self.processes_list, footer=self.buttons)

        self._tree = True
        self._paused = False
        self._thread = False
        self.loop = urwid.MainLoop(
            self.main_widget,
            self.palette,
            unhandled_input=self.handle_input,
        )
        self.refresh(None, None)

    def refresh(self, _1, _2):
        if not self._paused:
            self.processes.update(custom_process_filter, self._thread)
        self.processes_list.refresh(tree=self._tree)
        self.loop.set_alarm_in(self.refresh_rate_sec, self.refresh)

    def start(self):
        self.loop.run()

    def pause(self, key=None):
        text_map = {False: " Pause ", True: " Unpause "}
        self._paused = not self._paused
        text = text_map[self._paused]
        self.w_pause.set_label([('normal', 'F2 '), ('foot', text)])

    def quit(self, key=None):
        raise urwid.ExitMainLoop()

    def tree(self, key=None):
        text_map = {False: " Tree ", True: " Flat "}
        self._tree = not self._tree
        text = text_map[self._tree]
        self.w_tree.set_label([('normal', 'F3 '), ('foot', text)])

    def thread(self, key=None):
        text_map = {False: " Threads ", True: " No Threads "}
        self._thread = not self._thread
        text = text_map[self._thread]
        self.w_thread.set_label([('normal', 'F4 '), ('foot', text)])

    def handle_input(self, key):
        if type(key) == str:
            if key in ('q', 'Q', 'f10'):
                self.quit()
            if key == 'f1':
                self.display_help()
            if key == 'f2':
                self.pause()
            if key == 'f3':
                self.tree()
            if key == 'f4':
                self.thread()
        elif type(key) == tuple:
            pass


def shlex_join(argv):
    # TODO(eric.cousineau): Replace this with `shlex.join` when we exclusively
    # use Python>=3.8.
    return " ".join(map(shlex.quote, argv))


def get_command_or_name(p, *, short=True):
    args = p.cmdline()
    if len(args) == 0:
        return p.name()
    else:
        if short:
            args[0] = os.path.basename(args[0])
        return shlex_join(args)


def affinity_str(cpus, cpu_count=None):
    if cpu_count is None:
        cpu_count = psutil.cpu_count()
    cpus = list(sorted(cpus))
    if len(cpus) in [0, cpu_count]:
        return ""
    assert len(cpus) > 0
    # Try to compress contiguous ranges.
    cur = [cpus[0]]
    ranges = [cur]
    for cpu in cpus[1:]:
        if cpu == cur[-1] + 1:
            cur.append(cpu)
        else:
            cur = [cpu]
            ranges.append(cur)
    pieces = []
    for cur in ranges:
        if len(cur) == 1:
            pieces += [str(cur[0])]
        elif len(cur) == 2:
            pieces += [str(cur[0]), str(cur[1])]
        else:
            pieces += [f"{cur[0]}-{cur[-1]}"]
    inner = ",".join(pieces)
    return f"[{inner}]"


def process_iter(with_threads):
    """
    Like psutil.process_iter(), but add threads in.
    Returns (p, None) if process; returns (t, p) if thread.
    """
    for p in psutil.process_iter():
        yield p, None
        if not with_threads:
            continue
        # Include threads.
        for thread in p.threads():
            if thread.id == p.pid:
                continue
            try:
                t = psutil.Process(thread.id)
                yield t, p
            except psutil.NoSuchProcess:
                continue


def sched_str(sched):
    to_str = {
        os.SCHED_OTHER: "o",
        os.SCHED_BATCH: "b",
        os.SCHED_IDLE: "idle",
        os.SCHED_FIFO: "fifo",
        os.SCHED_RR: "rr",
    }
    suffix = ""
    if sched & os.SCHED_RESET_ON_FORK:
        suffix = " (reset)"
        sched &= ~os.SCHED_RESET_ON_FORK
    s = to_str[sched]
    return s


def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def human_sorting_key(text):
    return tuple(_tryint(s) for s in re.split(r"(\d+)", text))


def main():
    parser = argparse.ArgumentParser(prog='pytop')
    parser.add_argument(
        '-r', '--refresh_rate_sec', type=float, default=1.0,
        help='Refresh rate (sec)',
    )
    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s ' + __version__ + ' - ' + __copyright__,
    )

    args = parser.parse_args()

    app = Application(
        refresh_rate_sec=args.refresh_rate_sec,
    )
    app.start()
