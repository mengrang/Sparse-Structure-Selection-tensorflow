def update(index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if self.idx2name[index].startswith(self.lambda_name):
            # APG
            if state is not None:
                mom = state
                mom[:] *= self.momentum
                z = weight - lr * grad # equ 10
                z = self.soft_thresholding(z, lr * self.gamma)
                mom[:] = z - weight + mom # equ 11
                weight[:] = z + self.momentum * mom # equ 12
            else:
                assert self.momentum == 0.0