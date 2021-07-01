import torch
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QManager(object):

    def __init__(self, args, q_trace, q_batch):
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.traces_p = []
        self.traces_masks = []
        self.traces_logits = []

        self.lock = mp.Lock()

        self.q_trace = q_trace
        self.q_batch = q_batch
        self.args = args

    def listening(self):
        while True:
            trace = self.q_trace.get(block=True)
            self.traces_s.append(trace[0])
            self.traces_a.append(trace[1])
            self.traces_r.append(trace[2])
            self.traces_p.append(trace[3])
            self.traces_masks.append(trace[4])
            self.traces_logits.append(trace[5])

            if len(self.traces_s) > self.args.batch_size:
                self.produce_batch()

    def produce_batch(self):
        batch_size = self.args.batch_size
        res_s = self.traces_s[:batch_size]
        res_a = self.traces_a[:batch_size]
        res_r = self.traces_r[:batch_size]
        res_p = self.traces_p[:batch_size]
        res_masks = self.traces_masks[:batch_size]
        res_logits = self.traces_logits[:batch_size]

        del self.traces_s[:batch_size]
        del self.traces_a[:batch_size]
        del self.traces_r[:batch_size]
        del self.traces_p[:batch_size]
        del self.traces_masks[:batch_size]
        del self.traces_logits[:batch_size]

        # stack batch and put
        self.q_batch.put((torch.stack(res_s, dim=0).to(device), torch.stack(res_a, dim=0).to(device),
                          torch.stack(res_r, dim=0).to(device), torch.stack(res_p, dim=0).to(device),
                          torch.stack(res_masks, dim=0).to(device), torch.stack(res_logits, dim=0).to(device)))





