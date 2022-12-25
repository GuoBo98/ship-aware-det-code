from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class PrintEpoch(Hook):
    def __init__(self, a):
        pass
    
    def before_run(self, runner):
        print('#################################')

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        print('#################################')

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
    
