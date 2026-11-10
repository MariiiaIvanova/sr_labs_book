import os
import sys
import json
import time
import subprocess
import numpy as np
np.float = float

sys.path.append(os.path.realpath('..'))

import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def sync():

    if DEVICE == 'cuda':
        torch.cuda.synchronize()


def case_mixing():
    # mHC mixing on the shape of layer1: (128, 32, 40, 202), forward + backward

    from ResNetBlocksMHC import MHCShortcut

    torch.manual_seed(0)
    m = MHCShortcut(32)
    with torch.no_grad():
        m.theta.add_(torch.randn(4, 4))          # a non-trivial W

    x_cpu = torch.randn(128, 32, 40, 202)
    W = m.mixing_matrix().detach()
    y_ref = torch.einsum('ij,bjl->bil', W, x_cpu.reshape(128, 4, -1)).reshape(x_cpu.shape)

    m = m.to(DEVICE)
    x = x_cpu.to(DEVICE).requires_grad_(True)
    y = m(x)
    y.backward(torch.ones_like(y))
    sync()

    err = (y.detach().cpu() - y_ref).abs().max().item()
    grad_ok = bool(torch.isfinite(m.theta.grad).all().item()) and m.theta.grad.abs().sum().item() > 0

    return {'max_abs_diff_vs_cpu': err, 'theta_grad_ok': grad_ok}


def case_model(block_name, batch_size=128, max_frames=200, n_steps=10, nClasses=1211):
    # One training step repeated n_steps times, as in train_network

    from exercises_blank import ResNet, MainModel
    from ResNetBlocks import BasicBlock
    from ResNetBlocksMHC import BasicBlockMHC
    from LossFunction import AAMSoftmaxLoss
    from Optimizer import SGDOptimizer

    block = BasicBlockMHC if block_name == 'mhc' else BasicBlock

    model = ResNet(block, layers=[3, 4, 6, 3], activation=nn.ReLU, num_filters=[32, 64, 128, 256],
                   nOut=512, encoder_type='SP', n_mels=40, log_input=True)
    main_model = MainModel(model, AAMSoftmaxLoss(nOut=512, nClasses=nClasses, margin=0.35, scale=32.0)).to(DEVICE)
    optimizer = SGDOptimizer(main_model.parameters(), lr=0.01, weight_decay=0)
    main_model.train()

    data = torch.randn(1, batch_size, max_frames*160 + 240).to(DEVICE)
    label = torch.randint(0, nClasses, (batch_size,)).to(DEVICE)

    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    times = []
    for step in range(n_steps):
        sync()
        t = time.time()

        optimizer.zero_grad()
        nloss, prec1 = main_model(data, label)
        nloss.backward()
        optimizer.step()

        sync()
        times.append(time.time() - t)

    peak = torch.cuda.max_memory_allocated()/2**30 if DEVICE == 'cuda' else float('nan')
    step_time = float(np.median(times[2:])) if n_steps > 3 else float(np.median(times))

    return {'peak_gb': peak, 'step_s': step_time, 'loss_finite': bool(np.isfinite(nloss.item()))}


def run_case(name, launch_blocking, extra_args):
    # Runs one test in a separate python process and returns its result (or the error text)

    env = dict(os.environ)
    if launch_blocking:
        env['CUDA_LAUNCH_BLOCKING'] = '1'

    proc = subprocess.run([sys.executable, os.path.abspath(__file__), '--case', name] + extra_args,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            return True, json.loads(line[len('RESULT '):])

    return False, (proc.stderr.strip().splitlines() or ['no output'])


def print_error(lines):

    # the most informative part of a traceback: the last 15 lines
    for line in lines[-15:]:
        print('    ' + line)


if __name__ == "__main__":

    # Child process: run one test and print its result
    if len(sys.argv) >= 3 and sys.argv[1] == '--case':
        name = sys.argv[2]
        small = '--small' in sys.argv

        if name == 'mixing':
            res = case_mixing()
        else:
            res = case_model(name, batch_size=8 if small else 128, n_steps=3 if small else 10)

        print('RESULT ' + json.dumps(res))
        sys.exit(0)

    # Parent process
    extra = ['--small'] if '--small' in sys.argv else []     # --small: tiny batch, only for testing this script on CPU

    if DEVICE == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print('GPU: {:s}, memory {:.1f} GB, PyTorch {:s}, CUDA {:s}, cuDNN {:s}'.format(
            props.name, props.total_memory/2**30, torch.__version__, str(torch.version.cuda), str(torch.backends.cudnn.version())))
        try:
            apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
            print('Other processes on the GPU: {:s}'.format(apps if apps else 'none'))
        except OSError:
            pass
    else:
        print('CUDA is not available, running on CPU (memory is not measured)')

    all_ok = True

    # 1. mixing operation alone
    ok, res = run_case('mixing', True, extra)
    if ok:
        good = res['max_abs_diff_vs_cpu'] < 1e-4 and res['theta_grad_ok']
        all_ok &= good
        print('[{:s}] mHC mixing, shape (128, 32, 40, 202): max |GPU - CPU| = {:.2e}, gradient on Theta {:s}'.format(
            'OK' if good else 'FAIL', res['max_abs_diff_vs_cpu'], 'ok' if res['theta_grad_ok'] else 'BAD'))
    else:
        all_ok = False
        print('[FAIL] mHC mixing:')
        print_error(res)

    # 2-3. training step of both models
    results = {}
    for name, title in [('base', 'Base ResNet '), ('mhc', 'ResNet + mHC')]:
        ok, res = run_case(name, False, extra)

        if not ok:
            # repeat with CUDA_LAUNCH_BLOCKING=1 to see the exact operation that fails
            ok_b, res_b = run_case(name, True, extra)
            all_ok = False
            print('[FAIL] {:s}: error (traceback with CUDA_LAUNCH_BLOCKING=1):'.format(title))
            print_error(res_b if not ok_b else res)
            continue

        results[name] = res
        print('[OK] {:s}: peak memory {:.2f} GB, {:.3f} s/step, loss finite: {}'.format(title, res['peak_gb'], res['step_s'], res['loss_finite']))

    if 'base' in results and 'mhc' in results:
        t0, t1 = results['base']['step_s'], results['mhc']['step_s']
        print('mHC vs base: {:+.1f}% time per step, {:+.2f} GB peak memory'.format(100*(t1 - t0)/t0, results['mhc']['peak_gb'] - results['base']['peak_gb']))
        print('1047 steps/epoch x 40 epochs (compute only, data loading not included): base {:.1f} h, mHC {:.1f} h'.format(t0*1047*40/3600, t1*1047*40/3600))

    print('ALL CHECKS PASSED, training can be started' if all_ok else 'SOME CHECKS FAILED, send this output')
