"""Keep a long Colab run alive across sessions (docs/SCALE_TEST_14B.md, addendum of 9 Oct).

Colab's free GPU time runs out; the machine is then wiped, /content with it, and a run that was halfway through is gone. The runners (`run_scope_regime`, `run_long`) already
write one JSON line per finished task or episode and carry on from the lines they find. What was missing is somewhere that survives the wipe: Google Drive.
`Store` keeps the working files in /content/out (fast, plain files) and, after every finished episode, copies the file to a folder on Drive (through a temporary name and an
atomic rename, so a session that dies while copying leaves the last good copy). In a new session `restore` brings the file back and the runner continues from it.
Nothing here touches the model, the tasks, the policies or the reading rules: the test is the same, only its progress survives."""
import os, shutil


class Store:
    def __init__(self, local_dir, remote_dir=None):
        self.local_dir, self.remote_dir = local_dir, remote_dir
        os.makedirs(local_dir, exist_ok=True)
        if remote_dir: os.makedirs(remote_dir, exist_ok=True)

    @property
    def persistent(self) -> bool: return self.remote_dir is not None

    def path(self, name: str) -> str: return os.path.join(self.local_dir, name)

    def restore(self, name: str) -> str:
        """Bring `name` back from the remote folder if it is there and the local copy is missing or smaller. Returns 'restored', 'kept local' or 'nothing'."""
        local = self.path(name)
        if not self.remote_dir: return "kept local" if os.path.exists(local) else "nothing"
        remote = os.path.join(self.remote_dir, name)
        if os.path.exists(remote) and (not os.path.exists(local) or os.path.getsize(remote) > os.path.getsize(local)):
            shutil.copyfile(remote, local); return "restored"
        return "kept local" if os.path.exists(local) else "nothing"

    def persist(self, name: str) -> bool:
        """Copy `name` to the remote folder (temporary name, then atomic rename). False if there is no remote folder or no local file."""
        local = self.path(name)
        if not self.remote_dir or not os.path.exists(local): return False
        remote = os.path.join(self.remote_dir, name); tmp = remote + ".tmp"
        shutil.copyfile(local, tmp); os.replace(tmp, remote); return True

    def progress(self, name: str, inner=None):
        """A progress callback for the runners: call `inner` (the counter), then save the file after every finished unit."""
        def cb(i, n):
            if inner: inner(i, n)
            self.persist(name)
        return cb


def open_store(local_dir="/content/out", subdir="axi_14b", mount_point="/content/drive"):
    """Mount Google Drive (a pop-up asks permission) and return a Store whose remote folder is MyDrive/<subdir>. Without Drive, the Store works but keeps nothing across sessions."""
    try:
        from google.colab import drive
        drive.mount(mount_point)
        remote = os.path.join(mount_point, "MyDrive", subdir)
        st = Store(local_dir, remote); print(f"Drive connected. Progress is saved to MyDrive/{subdir} after every finished episode."); return st
    except Exception as e:                                                                # no Drive, or permission refused
        print("Drive NOT connected (" + type(e).__name__ + "). The run still works, but if Colab stops, the progress is lost.")
        return Store(local_dir, None)
