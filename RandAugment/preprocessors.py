import torch
import torchvision
from torch import nn
from abc import abstractmethod, ABCMeta
from torchvision import transforms
from RandAugment import google_augmentations, augmentations
from RandAugment.networks.convnet import SeqConvNet
from RandAugment.common import sigmax, log_sigmax

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
                 label_smoothing_rate, usemodel_Dout_imgdims=(False, 10, None), dist_functions=(torch.softmax, torch.log_softmax)):
        super().__init__()
        self.dist, self.log_dist = dist_functions
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
        self.p_op = self.dist(self.op_logits, 0)
        self.log_p_op = self.log_dist(self.op_logits, 0)
        sampled_op_idxs = torch.multinomial(self.p_op, num_samples, replacement=True)
        hidden = self.op_embs[sampled_op_idxs]
        if self.q_residual:
            hidden += self.q
        scale_logits = hidden @ self.scale_embs.t()
        p_scale = self.dist(scale_logits, 1)
        log_p_scale = self.log_dist(scale_logits, 1)
        sampled_scales = torch.multinomial(p_scale, 1, replacement=True).squeeze()
        self.logps = self.log_p_op[sampled_op_idxs] + log_p_scale[torch.arange(log_p_scale.shape[0]), sampled_scales]
        if self.label_smoothing_rate:
            self.logps = (self.log_p_op.mean() * len(sampled_op_idxs) + log_p_scale.mean(1).sum(
                0)) * self.label_smoothing_rate \
                         + self.logps * (1. - self.label_smoothing_rate)
        return sampled_op_idxs.cpu(), sampled_scales.cpu()


class LearnedPreprocessorRandaugmentSpace(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, bs, val_bs, entropy_alpha, scale_entropy_alpha=0.,
                 importance_sampling=False, cutout=0, normalize_reward=True, model_for_online_tests=None,
                 D_out=10, q_zero_init=True, q_residual=False, scale_embs_zero_init=False, label_smoothing_rate=0.,
                 sigmax_dist=False, use_images_for_sampler=False, **kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')
        assert not use_images_for_sampler

        img_dims = dataset_info['img_dims']
        self.num_transforms = google_augmentations.num_augmentations()
        self.num_scales = google_augmentations.PARAMETER_MAX + 1
        self.standard_cifar_preprocessor = StandardCIFARPreprocessor(dataset_info, cutout=cutout,
                                                                     device=self.device)
        self.model = model_for_online_tests

        self.augmentation_sampler = AugmentationSampler(hidden_dimension, self.num_transforms, self.num_scales,
                                                        q_residual, q_zero_init, scale_embs_zero_init,
                                                        label_smoothing_rate,
                                                        usemodel_Dout_imgdims=(self.model is not None, D_out, img_dims),
                                                        dist_functions=(sigmax, log_sigmax) if sigmax_dist
                                                        else (torch.softmax, torch.log_softmax)).to(self.device)
        self.agumentation_sampler_copies = []

        self.normalize_reward = normalize_reward
        self.optimizer = optimizer_creator(self.augmentation_sampler.parameters())
        self.entropy_alpha = entropy_alpha
        self.scale_entropy_alpha = scale_entropy_alpha
        self.importance_sampling = importance_sampling
        self.bs = bs
        self.val_bs = val_bs

    def forward(self, imgs, step):
        self.t = step
        if self.training:
            aug_sampler = deepcopy(self.augmentation_sampler)
            self.agumentation_sampler_copies.append(aug_sampler)
            sampled_op_idxs, sampled_scales = aug_sampler(self.bs, self.model)
            activated_transforms_for_batch = list(zip(sampled_op_idxs, sampled_scales)) + [(0, 1.) for _ in range(self.val_bs)]
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
        assert len(self.agumentation_sampler_copies) <= 2
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
            log_p_scale = aug_sampler.log_dist(scale_logits, 1)
            p_scale = aug_sampler.dist(scale_logits, 1)
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

    def reset_state(self):
        del self.agumentation_sampler_copies
        self.agumentation_sampler_copies = []

    def write_summary(self, aug_sampler, step):
        if step % 100 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                q = aug_sampler.compute_q()  # need to compute q here, because of print in first step
                op_logits = aug_sampler.op_embs @ q
                p = aug_sampler.dist(op_logits, 0)
                for i, p_ in enumerate(p):
                    self.summary_writer.add_scalar(f'PreprocessorWeights/p_{i}', p_, step)
                hidden = aug_sampler.op_embs
                if aug_sampler.q_residual:
                    hidden += q
                scale_logits = hidden @ aug_sampler.scale_embs.t()
                p_scale = aug_sampler.dist(scale_logits, 1)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'MaxScale/aug_{aug_idx}', torch.argmax(scale_dist), step)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'AvgScale/aug_{aug_idx}', scale_dist @ torch.arange(scale_dist.shape[-1],dtype=scale_dist.dtype,device=scale_dist.device), step)
                for i, p_ in enumerate(p_scale.mean(0)):
                    self.summary_writer.add_scalar(f'PreprocessorWeightsScale/p_{i}', p_, step)

class RandAugmentationSampler(nn.Module):
    def __init__(self, hidden_dimension, num_transforms, num_scales, possible_num_sequential_transforms, q_residual, q_zero_init, scale_embs_zero_init,
                 label_smoothing_rate, distribution_functions, use_images):
        super().__init__()
        self.dist, self.log_dist = distribution_functions
        self.op_embs = nn.Parameter(torch.normal(0., 1., (num_transforms, hidden_dimension), requires_grad=True))
        self.num_transforms_embs = nn.Parameter(torch.normal(0., 1., (len(possible_num_sequential_transforms), hidden_dimension), requires_grad=True))
        if scale_embs_zero_init:
            self.scale_embs = nn.Parameter(torch.zeros(num_scales, hidden_dimension, requires_grad=True))
        else:
            self.scale_embs = nn.Parameter(torch.normal(0., 1., (num_scales, hidden_dimension), requires_grad=True))
        self.label_smoothing_rate = label_smoothing_rate
        self.q_residual = q_residual
        self.use_images = use_images
        if use_images:
            self.convnet = SeqConvNet(hidden_dimension)
            if q_zero_init:
                with torch.no_grad():
                    self.convnet.final_fc.weight.zero_()
                    self.convnet.final_fc.bias.zero_()
        else:
            if q_zero_init:
                self.q = nn.Parameter(torch.zeros(hidden_dimension, requires_grad=True))
            else:
                self.q = nn.Parameter(torch.normal(0., 1., hidden_dimension, requires_grad=True))

        self.possible_num_sequential_transforms = possible_num_sequential_transforms

    def add_grad_of_copy(self, copy):
        # zero grad beforehand
        for p, p_copy in zip(self.parameters(), copy.parameters()):
            if p.grad is None:
                p.grad = p_copy.grad
            else:
                p.grad += p_copy.grad

    def compute_q(self, imgs):
        if self.use_images:
            image_batch = torch.stack([torchvision.transforms.ToTensor()(i) for i in imgs]).to(self.convnet.final_fc.weight.device)
            return self.convnet(image_batch)
        else:
            return self.q.unsqueeze(0).expand(len(imgs),-1)

    def forward(self, imgs):
        num_samples = len(imgs)
        self.q = self.compute_q(imgs)
        self.num_transforms_logits = self.q @ self.num_transforms_embs.t()
        self.p_num_transforms = self.dist(self.num_transforms_logits, -1)
        self.log_p_num_transforms = self.log_dist(self.num_transforms_logits, -1)
        sampled_transform_indices = torch.multinomial(self.p_num_transforms, 1, replacement=True).squeeze(1)
        sampled_num_transforms = self.possible_num_sequential_transforms[sampled_transform_indices]
        augmentation_inds = torch.randint(len(self.op_embs),(num_samples,len(self.num_transforms_embs)-1))
        augmentation_mask = torch.arange(len(self.num_transforms_embs)-1).unsqueeze(0).expand(num_samples,len(self.num_transforms_embs)-1).to(sampled_num_transforms.device) >= sampled_num_transforms.unsqueeze(1)
        augmentation_inds[augmentation_mask] = 0 # index of identity augmentation
        hidden = self.op_embs[augmentation_inds]
        if self.q_residual:
            hidden += self.q.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        scale_logits = hidden @ self.scale_embs.t()
        p_scale = self.dist(scale_logits, 2)
        log_p_scale = self.log_dist(scale_logits, 2)
        sampled_scales = torch.multinomial(p_scale.view(-1,p_scale.shape[2]), 1, replacement=True).view(p_scale.shape[:2])
        flat_log_p_scale = log_p_scale.view(-1,log_p_scale.shape[2])
        log_ps_of_sampled_scales = flat_log_p_scale[torch.arange(len(flat_log_p_scale)),sampled_scales.flatten()].view_as(sampled_scales)
        log_ps_of_sampled_scales[augmentation_mask] = 0.
        self.logps = self.log_p_num_transforms[torch.arange(len(self.log_p_num_transforms)),sampled_transform_indices] + log_ps_of_sampled_scales.sum(1)
        return augmentation_inds.cpu(), sampled_scales.cpu()


class LearnedRandAugmentPreprocessor(ImagePreprocessor):
    def __init__(self, dataset_info, hidden_dimension, optimizer_creator, bs, val_bs, entropy_alpha, scale_entropy_alpha=0.,
                 importance_sampling=False, cutout=0, normalize_reward=True, model_for_online_tests=None,
                 D_out=10, q_zero_init=True, q_residual=False, scale_embs_zero_init=False, label_smoothing_rate=0.,
                 possible_num_sequential_transforms=[1,2,3,4], sigmax_dist=False, use_images_for_sampler=False,
                 **kwargs):
        super().__init__(**kwargs)
        print('using learned preprocessor')

        img_dims = dataset_info['img_dims']
        self.num_transforms = google_augmentations.num_augmentations()
        self.num_scales = google_augmentations.PARAMETER_MAX + 1
        self.standard_cifar_preprocessor = StandardCIFARPreprocessor(dataset_info, cutout=cutout,
                                                                     device=self.device)
        self.model = model_for_online_tests

        self.augmentation_sampler = RandAugmentationSampler(hidden_dimension, self.num_transforms, self.num_scales, torch.tensor(possible_num_sequential_transforms),
                                                        q_residual, q_zero_init, scale_embs_zero_init,
                                                        label_smoothing_rate, (sigmax, log_sigmax) if sigmax_dist else (torch.softmax, torch.log_softmax),use_images_for_sampler).to(self.device)
        self.agumentation_sampler_copies = []
        self.bs = bs
        self.val_bs = val_bs

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
            sampled_op_idxs, sampled_scales = aug_sampler(imgs)
            activated_transforms_for_batch = [[(i,s) for i,s in zip(i_s,s_s)] for i_s,s_s in zip(sampled_op_idxs,sampled_scales)] + [[(0,1.)] for _ in range(self.val_bs)]
            self.write_summary(aug_sampler, step)
        else:
            activated_transforms_for_batch = [[(0, 1.)] for i in imgs]  # do not apply any augmentation when evaluating
        t_imgs = []
        google_augmentations.blend_images = imgs
        for i, (img, augs) in enumerate(zip(imgs, activated_transforms_for_batch)):
            for op, scale in augs:
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
        assert len(self.agumentation_sampler_copies) <= 2
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
            neg_entropy = torch.einsum('bt,bt->b',aug_sampler.p_num_transforms,(aug_sampler.log_p_num_transforms * (aug_sampler.p_num_transforms != 0.))).mean()
            loss += self.entropy_alpha * neg_entropy
        if self.scale_entropy_alpha:
            hidden = aug_sampler.op_embs
            if aug_sampler.q_residual:
                hidden = aug_sampler.op_embs + aug_sampler.q
            scale_logits = hidden @ aug_sampler.scale_embs.t()
            log_p_scale = aug_sampler.log_dist(scale_logits, 1)
            p_scale = aug_sampler.dist(scale_logits, 1)
            avg_neg_entropy = torch.einsum('bh,bh->b', p_scale, (
                        log_p_scale * (p_scale != 0.))).mean()  # here we could have logp = -inf and p = 0.
            loss += self.scale_entropy_alpha * avg_neg_entropy

        loss.backward()
        torch.nn.utils.clip_grad_value_(aug_sampler.parameters(), 5.)
        self.augmentation_sampler.zero_grad()
        self.augmentation_sampler.add_grad_of_copy(aug_sampler)
        self.optimizer.step()
        del aug_sampler

    def reset_state(self):
        del self.agumentation_sampler_copies
        self.agumentation_sampler_copies = []

    def write_summary(self, aug_sampler, step):
        if step % 10 == 0 and self.summary_writer is not None and self.training:
            with torch.no_grad():
                for i,p in enumerate(aug_sampler.p_num_transforms.mean(0)):
                    self.summary_writer.add_scalar(f'NumAugs/{i}', p, step)
                augmentation_inds = torch.arange(len(aug_sampler.op_embs)).unsqueeze(0).expand(len(aug_sampler.q),-1)
                hidden = aug_sampler.op_embs[augmentation_inds]
                if aug_sampler.q_residual:
                    hidden += aug_sampler.q.unsqueeze(1).expand(-1,hidden.shape[1],-1)
                scale_logits = hidden @ aug_sampler.scale_embs.t()
                p_scale = aug_sampler.dist(scale_logits, 2).mean(0)

                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'AvgScale/aug_{aug_idx}', scale_dist @ torch.arange(scale_dist.shape[-1],dtype=scale_dist.dtype,device=scale_dist.device), step)
                for aug_idx, scale_dist in enumerate(p_scale):
                    self.summary_writer.add_scalar(f'MaxScale/aug_{aug_idx}', torch.argmax(scale_dist), step)

