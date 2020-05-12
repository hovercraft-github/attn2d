from math import sqrt, pi, cos, ceil
from torch import optim
from torch.optim import lr_scheduler


def LRScheduler(opt, optimizer, last_epoch=-1):
    ref = opt['schedule']
    if ref == "early-stopping":
        if opt['criterion'] == "loss":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=opt['patience'],
                                                       factor=opt['decay_rate'],
                                                       verbose=True,
                                                       threshold=0.01,
                                                       min_lr=1e-5)
        elif opt['criterion'] == "perf":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode="max",
                                                       patience=opt['patience'],
                                                       factor=opt['decay_rate'],
                                                       verbose=True,
                                                       threshold=0.05)

    elif ref == "cosine-ep":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt['max_epochs'],
                                                   eta_min=opt.get('min_lr', 0))

    elif ref == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt['max_updates'],
                                                   eta_min=opt.get('min_lr', 0))
    elif ref == "shifted-cosine":
        scheduler = ShiftedCosine(optimizer,
                                  T_max=opt['max_updates'],
                                  cycles=opt['cycles'])

    elif ref == "plateau-cosine":
        scheduler = PlateauCosine(optimizer,
                                  T1=opt['T1'],
                                  T2=opt['T2'],
                                  eta1=opt['eta1'],
                                  eta2=opt['eta2'])

    elif ref == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt['decay_every'],
                                        gamma=opt['decay_rate'],
                                        last_epoch=last_epoch)
        # self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer.optimizer, self.anneal)
    elif ref == "step-iter":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt['decay_every'],
                                        gamma=opt['decay_rate'],
                                        last_epoch=last_epoch)

    elif ref == "inverse-square":
        scheduler = InverseSquareRoot(optimizer,
                                      warmup=opt['warmup'],
                                      last_epoch=last_epoch)

    elif ref == 'multi-step':
        milestones = list(opt['milestones'].split(','))
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones,
                                             gamma=opt['decay_rate'],
                                             last_epoch=last_epoch)
    else:
        raise ValueError('Unknown scheduler % s' % ref)
    scheduler.mode = ref
    return scheduler


class NAG(optim.Optimizer):
    def __init__(self, params,
                 lr=.25, momentum=0,
                 weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
        super(NAG, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        source: fairseq/optim/nag.py
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            lr_old = group.get('lr_old', lr)
            lr_correct = lr / lr_old

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = d_p.clone().zero_()

                buf = param_state['momentum_buffer']

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(momentum * momentum * lr_correct, buf)
                p.data.add_(-(1 + momentum) * lr, d_p)

                buf.mul_(momentum * lr_correct).add_(-lr, d_p)

            group['lr_old'] = lr

        return loss

class Optimizer(object):
    """
    Wrapper for the optimizer (fairseq style)
    """
    def __init__(self, opt, model):
        super().__init__()
        #  rmsprop | sgd | sgdmom | adagrad | adam
        ref = opt['solver'].lower()
        lr = opt['LR']['base']
        if isinstance(model, list):
            params = [{'params': m.parameters(),
                       'lr': lr}
                      for m in model]
        else:
            #params = [{'params': model.parameters(), 'lr': lr}]
            params = model.parameters()

        weight_decay = float(opt['weight_decay'])
        if ref == 'adam':
            optimizer = optim.Adam(params,
                                   lr=lr,
                                   betas=(opt['alpha'], opt['beta']),
                                   weight_decay=weight_decay,
                                   eps=float(opt['epsilon']),
                                   amsgrad=bool(opt.get('amsgrad', 0)))
        elif ref == 'sgd':
            optimizer = optim.SGD(params,
                                  lr=lr,
                                  momentum=opt.get('momentum', 0),
                                  dampening=opt.get('dampening', 0),
                                  weight_decay=weight_decay,
                                  nesterov=bool(opt.get('nesterov', 0)))

        elif ref.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params,
                                      lr=lr,
                                      #alpha=opt['alpha'],
                                      #eps=opt['epsilon'],
                                      weight_decay=opt['weight_decay']
                                      #momentum=opt.get('momentum', 0),
                                      #centered=False
                                      )
        elif ref.lower() == 'adagrad':
            optimizer = optim.Adagrad(params,
                                      lr=lr,
                                      lr_decay=opt.get('lr_decay', 0),
                                      weight_decay=opt['weight_decay'],
                                      initial_accumulator_value=0)
        elif ref.lower() == 'nag':
            optimizer = NAG(params,
                            lr=lr,
                            momentum=opt['momentum'],
                            weight_decay=opt['weight_decay']
                            )

        else:
            raise ValueError('Unknown optimizer % s' % ref)

        self.optimizer = optimizer

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load(self, state_dict):
        """Load an optimizer state dict. """
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        """Performs a single optimization step."""
        return self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        return self.optimizer.zero_grad()


    def require_grad(self):
        """Set requires_grad true for all params"""
        for p in self.optimizer.param_groups:
            if isinstance(p, dict):
                for pp in p['params']:
                    pp.requires_grad = True

        
class InverseSquareRoot(lr_scheduler._LRScheduler):
    """
    Follow the schedule of Vaswani et al. 2017
    """

    def __init__(self, optimizer,
                 warmup=4000, last_epoch=-1):
        self.warmup = warmup
        super(InverseSquareRoot, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch + 1
        scale_factor = min(1 / sqrt(it), it / self.warmup ** 1.5)
        return [base_lr * scale_factor
                for base_lr in self.base_lrs]


class ShiftedCosine(lr_scheduler._LRScheduler):

    """
    Similar to cosine
    """

    def __init__(self, optimizer, cycles, T_max, last_epoch=-1):
        self.T_max = T_max
        self.cycle_duration = ceil(T_max / cycles)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        scale_factor = 1/2 * (cos(pi * (self.last_epoch % self.cycle_duration) / self.cycle_duration) + 1)
        return [base_lr * scale_factor
                for base_lr in self.base_lrs]


class PlateauCosine(lr_scheduler._LRScheduler):

    """
    Steep decrease then ~ plateau
    [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                                    for base_lr in self.base_lrs]
    """

    def __init__(self, optimizer, T1, T2, eta1, eta2, last_epoch=-1):
        self.T1 = int(T1)
        self.T2 = int(T2)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.T1:
            # use a wider period
            return [self.eta2 + (self.eta1 - self.eta2) *
                    (1 + cos(pi * self.last_epoch / self.T2)) / 2
                    for base_lr in self.base_lrs]

        # very steep
        return [self.eta1 + (base_lr - self.eta1) *
                (1 + cos(pi * self.last_epoch / self.T1)) / 2
                for base_lr in self.base_lrs]



     
