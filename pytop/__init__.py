""" pytop.py: Htop copycat implemented in Python. """

__author__ = 'Andrii Oshtuk, Eric Cousinea'
__copyright__ = '(C) 2021 ' + __author__
__license__ = "MIT"
__version__ = '1.1.0'

import argparse
from collections import namedtuple
from datetime import timedelta
import shlex
import time

import psutil
import urwid


def shlex_join(argv):
    # TODO(eric.cousineau): Replace this with `shlex.join` when we exclusively
    # use Python>=3.8.
    return " ".join(map(shlex.quote, argv))


class Process:
    """
        Information about running process with PID.
        .. PROC(5)
            http://man7.org/linux/man-pages/man5/proc.5.html
        """

    def __init__(self, pid):
        assert isinstance(pid, int)
        self.pid = pid

        self.user = None
        self.status = None
        self.cpu_percent = 0.0
        self.time_sec = 0.0
        self.command = ''

    def update(self, p):
        """Retrieves actual process statistics from /proc/[pid]/ subdirectory."""
        assert isinstance(p, psutil.Process)
        self.command = p.name() #shlex_join(p.cmdline())
        self.user = p.username()
        self.niceness = p.nice()
        self.cpu_percent = p.cpu_percent()
        self.status = p.status()
        self.time_sec = time.time() - p.create_time()
        self.num_threads = p.num_threads()
        self.cpu_affinity = p.cpu_affinity()

    @property
    def time(self):
        """Time running, as string."""
        d = timedelta(seconds=float(self.time_sec))

        hours, remainder = divmod(d.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0.0:
            return '{hours}h:{minutes}m:{seconds}s'
        else:
            return '{minutes}m:{seconds:.2f}s'


def affinity_str(cpus, cpu_count=None):
    if cpu_count is None:
        cpu_count = psutil.cpu_count()
    cpus = list(sorted(cpus))
    if len(cpus) == cpu_count:
        return ""
    # Try to compress contiguous ranges.
    if len(cpus) > 1:
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
        inner = ", ".join(pieces)
        return f"[{inner}]"
    else:
        return str(cpus)


class ProcessesController:
    def __init__(self):
        self._process_map = dict()

    def update(self):
        actual_pids = set()
        for p in psutil.process_iter():
            with p.oneshot():
                actual_pids.add(p.pid)
                if p.pid not in self._process_map:
                    process = Process(p.pid)
                    self._process_map[p.pid] = process
                else:
                    process = self._process_map[p.pid]
                process.update(p)
        # Remove old.
        obsolete_pids = set(self._process_map.keys()) - actual_pids
        for pid in obsolete_pids:
            del self._process_map[pid]

    @property
    def processes(self):
        return list(self._process_map.values())

    @property
    def processes_number(self):
        return len(self._process_map)

    @property
    def running_processes_number(self):
        # TODO(eric): Is this supposed to be non-zombie procs? Or num threads?
        return self.processes_number

    @property
    def processes_pid(self):
        return [int(process.pid) for process in self._processes]


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

        self.cols = [
            ProcessColumn("USER", "{:<11}", lambda pr: f"{pr.user[:9]}"),
            ProcessColumn("PID", "{:<6}", lambda pr: f"{pr.pid}"),
            ProcessColumn("CPU%", "{:<10}", lambda pr: f"{pr.cpu_percent}"),
            ProcessColumn("NLWP", "{:<6}", lambda pr: f"{pr.num_threads}"),
            ProcessColumn("Afty", "{:<20}", lambda pr: affinity_str(pr.cpu_affinity)),
            ProcessColumn("Command", "{:<30}", lambda pr: pr.command[:30]),
        ]
        header_text = ""
        for col in self.cols:
            header_text += col.fmt.format(col.name)
        self.header = urwid.Text(('table_header', header_text))

        self.entries = urwid.SimpleFocusListWalker([])
        self.table_view = urwid.ListBox(self.entries)
        self.table_widget = urwid.Frame(self.table_view, header=self.header)

        self.refresh()

        urwid.WidgetWrap.__init__(self, self.table_widget)

    def _entry_key(self, entry):
        return -entry.pr.cpu_percent

    def _process_row_text(self, pr):
        result = []
        for col in self.cols:
            result.append(col.fmt.format(col.value(pr)))
        return result

    def refresh(self):
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
        self.entries.sort(key=self._entry_key)
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
        self.processes.update()

        # buttons
        help_ = urwid.Button([('normal', 'F1'), ('foot', 'Help')])
        urwid.connect_signal(help_, 'click', self.help)
        pause = urwid.Button([('normal', 'F2'), ('foot', 'Pause')])
        self._w_pause = pause
        urwid.connect_signal(pause, 'click', self.pause)
        quit = urwid.Button([('normal', 'F10'), ('foot', 'Quit')])
        urwid.connect_signal(quit, 'click', self.quit)

        # widgets
        self.buttons = urwid.Columns([help_, pause, quit])
        self.processes_list = ProcessPanel(self.processes)
        self.main_widget = urwid.Frame(self.processes_list, footer=self.buttons)

        self._paused = False
        self.loop = urwid.MainLoop(
            self.main_widget,
            self.palette,
            unhandled_input=self.handle_input,
        )
        self.loop.set_alarm_in(self.refresh_rate_sec, self.refresh)

    def refresh(self, loop, data):
        if not self._paused:
            self.processes.update()
        self.processes_list.refresh()
        self.loop.set_alarm_in(self.refresh_rate_sec, self.refresh)

    def start(self):
        self.loop.run()

    def help(self, key):
        self.display_help()

    def pause(self, key):
        text_map = {False: "Pause", True: "Unpause"}
        self._paused = not self._paused
        text = text_map[self._paused]
        self._w_pause.set_label([('normal', 'F2'), ('foot', text)])

    def quit(self, key):
        raise urwid.ExitMainLoop()

    def handle_input(self, key):
        if type(key) == str:
            if key in ('q', 'Q'):   #TODO(AOS) Remove in final version
                self.quit()
            if key == 'f1':
                self.display_help()
            if key == 'f2':
                self.pause(key)
            if key == 'f10':
                self.handle_f10_buton(key)
            else:
                self.loop.widget = self.main_widget
        elif type(key) == tuple:
            pass

    def display_help(self):
        help_txt = \
            f"""
        Pytop {__version__} - {__copyright__}
        Released under the {__license__}.

        Pytop is the htop copycat implemented in Python.

        usage: pytop [-h] [-v]

        optional arguments:
            -h, --help     show this help message and exit
            -v, --version  show program's version number and exit
        """
        self.help_txt = urwid.Text([('normal', help_txt),
                                    ('fields_names', '\nPress any key to return')],
                                    align='left')
        fill = urwid.Filler(self.help_txt, 'top')
        self.loop.widget = fill


def main():
    """ Returns script options parsed from CLI arguments."""
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
