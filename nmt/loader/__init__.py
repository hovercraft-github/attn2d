from .dataloader import textDataLoader


def ReadData(params, jobname):
    ddir = params['dir']
    src = params['src']
    trg = params['trg']
    dataparams = {'h5': "%s/%s.%s" % (ddir, src, "h5"),
                  'infos': "%s/%s.%s" % (ddir, src, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_src_length'],
                  'src': params['src']
                  #'type': params['type'],
                  #'input_dim': params['input_dim']
                 }
    src_loader = textDataLoader(dataparams, jobname=jobname)

    dataparams = {'h5': "%s/%s.%s" % (ddir, trg, "h5"),
                  'infos': "%s/%s.%s" % (ddir, trg, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_trg_length']
                 }

    trg_loader = textDataLoader(dataparams, jobname=jobname)
    return src_loader, trg_loader


def ReadRawData(params, h5path, jobname):
    ddir = params['dir']
    src = params['src']
    trg = params['trg']
    dataparams = {'h5': "%s.%s.h5" % (h5path, src),
                  'infos': "%s/%s.%s" % (ddir, src, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_src_length'],
                  "raw": True
                 }
    src_loader = textDataLoader(dataparams, jobname=jobname)

    dataparams = {'h5': "%s.%s.h5" % (h5path, trg),
                  'infos': "%s/%s.%s" % (ddir, trg, "infos"),
                  'batch_size': params['batch_size'],
                  'max_length': params['max_trg_length'],
                  "raw" : True
                 }

    trg_loader = textDataLoader(dataparams, jobname=jobname)
    return src_loader, trg_loader



