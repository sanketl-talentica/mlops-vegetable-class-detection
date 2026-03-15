# DVC Quick Reference

## Setup
```bash
dvc init                                              # initialize DVC in project (already done)
dvc remote add -d local_remote /tmp/dvcstore          # set local storage
```

## Running Pipeline
```bash
dvc repro                                             # run only changed stages
dvc repro --force                                     # force run all stages
```

## Tracking Data/Models
```bash
dvc push                                              # save data & models to remote
dvc pull                                              # fetch data & models from remote
dvc status                                            # check what's changed
```

## Metrics
```bash
dvc metrics show                                      # show current metrics.json values
dvc metrics diff                                      # compare metrics between runs
```

## Experiment Tracking
```bash
dvc params diff                                       # see what params changed since last run
dvc dag                                               # visualize your pipeline as a graph
```

## Day-to-Day Workflow
1. Make a change (data/config/code)
2. `dvc repro` — reruns only what's affected
3. `dvc metrics show` — check if accuracy improved
4. `dvc push` — save everything
