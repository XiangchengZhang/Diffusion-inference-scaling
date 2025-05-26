import os
import sys
import os.path as osp
import json

INFO = 20

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError
    
class SeqWriter:
    def writeseq(self, seq):
        raise NotImplementedError
    
class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))
    
def get_wandb_expr_info(args):
    return {k: v for k, v in vars(args).items() if \
            type(v) in [str, int, float, bool]}


class BaseLogger:
    """The logging system of the unified training-free guidance pipeline.
    Loggers are initialized in get_config() automatically.
    To use the logger, simply `import logger` and call `logger.log()` with the message you want to log.
    Golbally useful functions like `log_samples()` and `log_metrics()` are also provided.
    """
    def __init__(self, args, output_formats):
        self.level = INFO
        self.output_formats = output_formats
        self.logging_dir = args.logging_dir
        self.wandb = args.wandb
        if args.wandb:
            # lazy import wandb
            import wandb
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity
            )

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

    def log_samples(self, samples):
        # This will be automatically called after the use of instance-wise `log_samples`
        with open(os.path.join(self.logging_dir, "finished_sampling"), 'w') as f:
            f.write("\n")
    
    def load_samples(self, fname):
        raise NotImplementedError
        
    def log_metrics(self, metrics, save_json=False, suffix=''):
        self.log(metrics)

        if save_json:
            save_name = 'metrics.json' if suffix == '' else f'metrics_{suffix}.json'
            with open(osp.join(self.logging_dir, save_name), 'w') as f:
                json.dump({k:float(v) for k, v in metrics.items()}, f)
        
        if self.wandb:
            if suffix != '':
                metrics = {f'{k}_{suffix}': v for k, v in metrics.items()}
            self.wandb_run.log(metrics)
