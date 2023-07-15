import torch
import torch.nn.functional as F

class CorrBlock:
    # for disparity, i.e. stereo matching
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # generate grid index
        b, _, h, w = fmap1.shape
        self.coord_w = torch.arange(w).view(w, 1, 1, 1).repeat(b*h, 1, radius*2+1, 1).to(fmap1.device).float()
        self.coord_h = torch.ones_like(self.coord_w) * (-1.0)
        # all pairs correlation
        # [B*H*W, 1, 1, W]
        corr = CorrBlock.corr(fmap1, fmap2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, (1, 2), stride=(1, 2))
            self.corr_pyramid.append(corr)

    def __call__(self, disp):
        r = self.radius
        b, c, h, w = disp.shape
        assert c == 1, "{} got".format(c)

        # left image's disparity map
        coord_w = self.coord_w.float() - disp.permute(0, 2, 3, 1).contiguous().reshape(b*h*w, 1).view(b*h*w, 1, 1, 1)
        # [B*H*W, 1, 1, 2]
        coords = torch.cat((coord_w, self.coord_h), dim=-1)

        out_pyramid = []
        for i in range(self.num_levels):
            # [B*H*W, 1, 1, W]
            corr = self.corr_pyramid[i]
            # [1, 1, 2*r+1, 1]
            delta_w = torch.linspace(-r, r, 2*r+1).to(disp.device).view(1, 1, 2*r+1, 1)

            coords_lvl = coords / 2**i
            # [B*H*W, 1, 2*r+1, 2]
            coords_lvl[:, :, :, 0:1] = 2 * (coords_lvl[:, :, :, 0:1] + delta_w) / (w - 1) - 1

            # [B*H*W, 1, 1, 2*r+1]
            corr = F.grid_sample(corr, coords_lvl, mode='bilinear', padding_mode='zeros')
            # [B, H, W, 2*r+1]
            corr = corr.reshape(b, h, w, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        # [B, 4*(2*r+1), H, W]
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        # [B, H, W, C]
        fmap1 = fmap1.permute(0, 2, 3, 1)
        # [B, H, C, W]
        fmap2 = fmap2.permute(0, 2, 1, 3)

        # [B, H, W, W]
        corr = torch.matmul(fmap1, fmap2)
        # [B*H*W, 1, 1, W]
        corr  = corr.reshape(batch*ht*wd, wd).unsqueeze(1).unsqueeze(1)
        return corr / torch.sqrt(torch.tensor(dim).float())



class FlowCorrBlock:
    # for optical flow
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = FlowCorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        x2 = torch.matmul(fmap1.transpose(1, 2), fmap1)
        y2 = torch.matmul(fmap2.transpose(1, 2), fmap2)
        corr = (x2 - 2*corr + y2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

    @staticmethod
    def init_flow(size, device, flow_init=None):
        assert len(size) == 4, "Excepted size with [B, C, H, W], but {} got".format(size)
        b, c, h, w = size


        """ construct pixel coordination in an image"""
        # [1, H, W]  copy 0-width for h times  : x coord
        x_range = torch.arange(0, w, device=device, dtype=torch.float).view(1, 1, 1, w).expand(b, 1, h, w)
        # [1, H, W]  copy 0-height for w times : y coord
        y_range = torch.arange(0, h, device=device, dtype=torch.float).view(1, 1, h, 1).expand(b, 1, h, w)
        # [b, 2, h, w]
        pixel_coord = torch.cat((x_range, y_range), dim=1)

        ref_coord = pixel_coord.detach()
        tgt_coord = pixel_coord.detach()
        if flow_init is not None:
            tgt_coord = tgt_coord + flow_init

        return ref_coord, tgt_coord

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


if __name__ == '__main__':
    print("Test CorrBlock...")
    import time

    mode = 'flow'
    # mode = 'stereo'

    if mode == 'flow':
        iters = 50
        scale = 16
        B, C, H, W = 1, 80, 384, 1248
        device = 'cuda:0'

        prev = torch.randn(B, C, H//scale, W//scale, device=device)
        curr = torch.randn(B, C, H//scale, W//scale, device=device)

        ref_coord, tgt_coord = FlowCorrBlock.init_flow(size=(B, C, H//scale, W//scale), device=device, flow_init=None)
        coords = tgt_coord

        start_time = time.time()

        for i in range(iters):
            with torch.no_grad():
                corr_fn = FlowCorrBlock(prev, curr, num_levels=4, radius=4)
                cost = corr_fn(coords)

                torch.cuda.synchronize(device)
        end_time = time.time()
        avg_time = (end_time - start_time) / iters


        print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('FlowCorrBlock', avg_time * 1000, (1 / avg_time)))

    elif mode=='stereo':
        iters = 50
        scale = 4
        B, C, H, W = 1, 32, 384, 1248
        device = 'cuda:0'

        left = torch.randn(B, C, H // scale, W // scale, device=device)
        right = torch.randn(B, C, H // scale, W // scale, device=device)

        disp = torch.randn(B, 1, H//scale, W//scale, device=device) * 192

        start_time = time.time()

        for i in range(iters):
            with torch.no_grad():
                corr_fn = CorrBlock(left, right, num_levels=4, radius=4)
                cost = corr_fn(disp)

                torch.cuda.synchronize(device)
        end_time = time.time()
        avg_time = (end_time - start_time) / iters

        print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format('CorrBlock', avg_time * 1000,
                                                                                (1 / avg_time)))

    print("Done!")

"""
RAFT Flow: at scale=8, reference forward once takes 2.1072ms, i.e. 474.56fps 
RAFT Flow: at scale=8, reference forward once takes 1.1996ms, i.e. 833.65fps 
RAFT Stereo: at scale=4, reference forward once takes 1.7301ms, i.e. 578.01fps 
"""
