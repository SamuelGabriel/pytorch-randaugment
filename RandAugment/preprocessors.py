import torch
import torchvision
from torch import nn
from abc import abstractmethod, ABCMeta
from torchvision import transforms
from RandAugment import google_augmentations, augmentations
from RandAugment.networks.convnet import SeqConvNet

from copy import deepcopy

class Preprocessor(nn.Module):
    def __init__(self, device='cpu:0', summary_writer=None):
        super().__init__()
        self.summary_writer = summary_writer
        self.device = device

    def has_parameters(self):
        p = self.parameters()
        try:
            first = next(p)
            return True
        except StopIteration:
            return False

class ImagePreprocessor(Preprocessor, metaclass=ABCMeta):
    """Objects of this class can even hold weights that are trained with the meta loss.
    """
    @abstractmethod
    def forward(self, imgs):
        """Receives a batch of PIL img and returns a tensor with bs x c x h x w format.
        Depending on the `training` flag, should either perform the transformation
        for training or test.
        The test transform should _not be stochastic_.

        Args:
            imgs (List[PIL]): the image as it is in the data set
        Returns:
            Float Tensor bs x c x h x w
        """
        pass

def compute_dataset_stats(ds):
    mean = 0.
    std = 0.
    for img,label in ds:
        t = torchvision.transforms.ToTensor()(img)
        if len(t.shape) == 2:
            t.unsqueeze_(0)
        data = t.view(t.shape[0], -1)
        mean += data.mean(1)
        std += data.std(1)
    return mean / float(len(ds)), std / float(len(ds))


class StandardCIFARPreprocessor(ImagePreprocessor):
    def __init__(self, dataset_info, cutout, **kwargs):
        super().__init__(**kwargs)
        img_dims = dataset_info['img_dims']
        channels = img_dims[0]
        image_size_hw = tuple(img_dims[1:])

        self.normalize = transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        self.rand_crop = transforms.RandomCrop(image_size_hw, padding=(4, 4), padding_mode='constant')
        self.maybe_mirror = transforms.RandomHorizontalFlip(p=.5)
        if cutout:
            self.eraser = augmentations.CutoutDefault(cutout)
        else:
            self.eraser = lambda x: x

    def forward(self, imgs, step):
        with torch.no_grad():
            img_dims = torchvision.transforms.ToTensor()(imgs[0]).shape
            batch_tensor = torch.zeros((len(imgs),) + tuple(img_dims))
            for i, img in enumerate(imgs):
                if self.training:
                    img = self.rand_crop(img)
                    img = self.maybe_mirror(img)
                t = torchvision.transforms.ToTensor()(img)
                t = self.normalize(t)
                if self.training:
                    t = self.eraser(t)
                batch_tensor[i] = t

            return batch_tensor.to(self.device)


class AugmentationSampler(nn.Module):
    def __init__(self, hidden_dimension, num_transforms, num_scales, q_residual, q_zero_init, scale_embs_zero_init,
                 label_smoothing_rate, usemodel_Dout_imgdims=(False, 10, None)):
        super().__init__()
        self.op_embs = nn.Parameter(torch.normal(0., 1., (num_transforms, hidden_dimension), requires_grad=True))
        if scale_embs_zero_init:
            self.scale_embs = nn.Parameter(torch.zeros(num_scales, hidden_dimension, requires_grad=True))
        else:
            self.scale_embs = nn.Parameter(torch.normal(0., 1., (num_scales, hidden_dimension), requires_grad=True))
        use_model, D_out, img_dims = usemodel_Dout_imgdims
        if use_model:
            self.input_image = nn.Parameter(torch.normal(0., 1., img_dims, requires_grad=True))
            self.post_processor = nn.Sequential(
                nn.Linear(D_out, hidden_dimension),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dimension, hidden_dimension)
            )
            if q_zero_init:
                w = self.post_processor[2].weight
                b = self.post_processor[2].bias
                torch.nn.init.zeros_(w)
                torch.nn.init.zeros_(b)
        else:
            if q_zero_init:
                self.q = nn.Parameter(torch.zeros(hidden_dimension, requires_grad=True))
            else:
                self.q = nn.Parameter(torch.normal(0., 1., hidden_dimension, requires_grad=True))
        self.label_smoothing_rate = label_smoothing_rate
        self.q_residual = q_residual

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def compute_q(self, model=None):
        if model is not None:
            model_trains = model.training
            model.eval()
            o = model(self.input_image.unsqueeze(0))
            if model_trains:
                model.train()
            return self.post_processor(o).squeeze(0)
        else:
            return self.q

    def forward(self, num_samples, model=None):
        self.q = self.compute_q(model)
        self.op_logits = self.op_embs @ self.q
        self.p_op = torch.softmax(self.op_logits, 0)
        self.log_p_op = torch.log_softmax(self.op_logits, 0)
        sampled_op_idxs = torch.multinomial(self.p_op, num_samples, replacement=True)
        hidden = self.op_embs[sampled_op_idxs]
        if self.q_residual:
            hidden += self.q
        scale_logits = hidden @ self.scale_embs.t()
        p_scale = torch.softmax(scale_logits, 1)
        log_p_scale = torch.log_softmax(scale_logits, 1)
        sampled_scales = torch.multinomial(p_scale, 1, replacement=True).squeeze()
        self.logps = self.log_p_op[sampled_op_idxs] + log_p_scale[torch.arange(log_p_scale.shape[0]), sampled_scales]
        if self.label_smoothing_rate:
            self.logps = (self.log_p_op.mean() * len(sampled_op_idxs) + log_p_scale.mean(1).sum(
                0)) * self.label_smoothing_rate \
                         + self.logps * (1. - self.label_smoothing_rate)
        return sampled_op_idxs.cpu(), sampled_scales.cpu()


class LearnedPreprocessorRandaugmentSpace(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, entropy_alpha, scale_entropy_alpha=0.,
                 importance_sampling=False, cutout=0, normalize_reward=True, model_for_online_tests=None,
                 D_out=10, q_zero_init=True, q_residual=False, scale_embs_zero_init=False, label_smoothing_rate=0.,
                 **kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')

        img_dims = dataset_info['img_dims']
        self.num_transforms = google_augmentations.num_augmentations()
        self.num_scales = google_augmentations.PARAMETER_MAX + 1
        self.standard_cifar_preprocessor = StandardCIFARPreprocessor(dataset_info, cutout=cutout,
                                                                     device=self.device)
        self.model = model_for_online_tests

        self.augmentation_sampler = AugmentationSampler(hidden_dimension, self.num_transforms, self.num_scales,
                                                        q_residual, q_zero_init, scale_embs_zero_init,
                                                        label_smoothing_rate, usemodel_Dout_imgdims=(
            self.model is not None, D_out, img_dims)).to(self.device)
        self.agumentation_sampler_copies = []

        self.normalize_reward = normalize_reward
        self.optimizer = optimizer_creator(self.augmentation_sampler.parameters())
        self.entropy_alpha = entropy_alpha
        self.scale_entropy_alpha = scale_entropy_alpha
        self.importance_sampling = importance_sampling

    def forward(self, imgs, step):
        self.t = step
        if self.training:
            aug_sampler = deepcopy(self.augmentation_sampler)
            self.agumentation_sampler_copies.append(aug_sampler)
            sampled_op_idxs, sampled_scales = aug_sampler(len(imgs), self.model)
            activated_transforms_for_batch = list(zip(sampled_op_idxs, sampled_scales))
            self.write_summary(aug_sampler, step)
        else:
            activated_transforms_for_batch = [(0, 1.) for i in imgs]  # do not apply any augmentation when evaluating
        t_imgs = []
        for i, (img, (op, scale)) in enumerate(zip(imgs, activated_transforms_for_batch)):
            img = google_augmentations.apply_augmentation(op, scale, img)
            t_imgs.append(img)
        if self.importance_sampling and self.training:
            w = 1. / (self.num_transforms * self.num_scales * torch.exp(aug_sampler.logps.detach()))
            return self.standard_cifar_preprocessor(t_imgs, step), w * (len(w) / torch.sum(w))
        return self.standard_cifar_preprocessor(t_imgs, step)

    def compute_weights(self, rewards):
        if self.normalize_reward:
            rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    def step(self, rewards):
        aug_sampler = self.agumentation_sampler_copies.pop(0)  # pops the oldest state first (queue-style)
        if self.t % 100 == 0 and self.summary_writer is not None and self.training:
            self.summary_writer.add_scalar(f'Alignment/AverageAlignment', rewards.mean(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MaxAlignment', rewards.max(), self.t)
            self.summary_writer.add_scalar(f'Alignment/MinAlignment', rewards.min(), self.t)

        with torch.no_grad():
            if self.importance_sampling:
                rewards *= 1. / ((self.num_transforms * self.num_scales) ** 2 * torch.exp(2 * self.logps))
            weights = self.compute_weights(rewards).detach().to(self.device)

        aug_sampler.zero_grad()
        loss = -aug_sampler.logps @ weights.detach() / float(len(weights))
        if self.entropy_alpha:
            neg_entropy = aug_sampler.p_op @ (
                        aug_sampler.log_p_op * (aug_sampler.p_op != 0.))  # here we could have logp = -inf and p = 0.
            loss += self.entropy_alpha * neg_entropy
        if self.scale_entropy_alpha:
            hidden = aug_sampler.op_embs
            if aug_sampler.q_residual:
                hidden = aug_sampler.op_embs + aug_sampler.q
            scale_logits = hidden @ aug_sampler.scale_embs.t()
            log_p_scale = torch.log_softmax(scale_logits, 1)
            p_scale = torch.softmax(scale_logits, 1)
            avg_neg_entropy = torch.einsum('bh,bh->b', p_scale, (
                        log_p_scale * (p_scale != 0.))).mean()  # here we could have logp = -inf and p = 0.
            loss += self.scale_entropy_alpha * avg_neg_entropy

        if self.model:
            # This is needed so that we do not write to the gradient buffers of the model
            for p in self.model.parameters():
                assert p.requires_grad
                p.requires_grad = False
        loss.backward()
        if self.model:
            for p in self.model.parameters():
                p.requires_grad = True
        torch.nn.utils.clip_grad_value_(aug_sampler.parameters(), 5.)
        self.augmentation_sampler.zero_grad()
        self.augmentation_sampler.add_grad_of_copy(aug_sampler)
        self.optimizer.step()
        del aug_sampler

    def write_summary(self, aug_sampler, step):
        if step % 100 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                q = aug_sampler.compute_q()  # need to compute q here, because of print in first step
                op_logits = aug_sampler.op_embs @ q
                p = torch.softmax(op_logits, 0)
                for i, p_ in enumerate(p):
                    self.summary_writer.add_scalar(f'PreprocessorWeights/p_{i}', p_, step)
                hidden = aug_sampler.op_embs
                if aug_sampler.q_residual:
                    hidden += q
                scale_logits = hidden @ aug_sampler.scale_embs.t()
                p_scale = torch.softmax(scale_logits, 1)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'MaxScale/aug_{aug_idx}', torch.argmax(scale_dist), step)
                for i, p_ in enumerate(p_scale.mean(0)):
                    self.summary_writer.add_scalar(f'PreprocessorWeightsScale/p_{i}', p_, step)
