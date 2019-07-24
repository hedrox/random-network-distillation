import tensorflow as tf
from baselines.common.distributions import make_pdtype
from collections import OrderedDict
from gym import spaces

def canonical_dtype(orig_dt):
    if orig_dt.kind == 'f':
        return tf.float32
    elif orig_dt.kind in 'iu':
        return tf.int32
    else:
        raise NotImplementedError

class StochasticPolicy(object):
    def __init__(self, scope, ob_space, ac_space):
        self.abs_scope = (tf.get_variable_scope().name + '/' + scope).lstrip('/')
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.pdtype = make_pdtype(ac_space)
        self.ph_new = tf.placeholder(dtype=tf.float32, shape=(None, None), name='new')
        self.ph_ob_keys = []
        self.ph_ob_dtypes = {}
        shapes = {}
        if isinstance(ob_space, spaces.Dict):
            assert isinstance(ob_space.spaces, OrderedDict)
            for key, box in ob_space.spaces.items():
                assert isinstance(box, spaces.Box)
                self.ph_ob_keys.append(key)
            # Keys must be ordered, because tf.concat(ph) depends on order. Here we don't keep OrderedDict
            # order and sort keys instead. Rationale is to give freedom to modify environment.
            self.ph_ob_keys.sort()
            for k in self.ph_ob_keys:
                self.ph_ob_dtypes[k] = ob_space.spaces[k].dtype
                shapes[k] = ob_space.spaces[k].shape
        else:
            #print(ob_space)
            box = ob_space
            assert isinstance(box, spaces.Box)
            self.ph_ob_keys = [None]
            self.ph_ob_dtypes = {None: box.dtype}
            shapes = {None: box.shape}

        self.ph_ob = OrderedDict([(k, tf.placeholder(canonical_dtype(self.ph_ob_dtypes[k]),
                                                     (None, None,) + tuple(shapes[k]),
                                                     name=('obs/{}'.format(k) if k is not None else
                                                           'obs'))) for k in self.ph_ob_keys])

        assert list(self.ph_ob.keys()) == self.ph_ob_keys, "\n{}\n{}\n".format(list(self.ph_ob.keys()), self.ph_ob_keys)
        ob_shape = tf.shape(next(iter(self.ph_ob.values())))
        self.sy_nenvs  = ob_shape[0]
        self.sy_nsteps = ob_shape[1]
        self.ph_ac = self.pdtype.sample_placeholder([None, None], name='ac')
        self.pd = self.vpred = self.ph_istate = None

    def rel_to_abs(self, X):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Converts tensor from relative to absolute indexing
        """
        # [B, Nh, L, 2L-1]
        B, Nh, L, _ = X.shape

        # Pad to shift from relative to absolute indexing
        B = B if B.value is not None else -1
        ph_pad = tf.placeholder(tf.float32, shape=[None, Nh, L, 1])
        col_pad = tf.zeros_like(ph_pad)
        X = tf.concat([X, col_pad], axis=3)
        flat_x = tf.reshape(X, [B, Nh, L * 2 * L])

        ph_flat_pad = tf.placeholder(tf.float32, shape=[None, Nh, L-1])
        flat_pad = tf.zeros_like(ph_flat_pad)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)

        # Reshape and slice out the padded elements.
        final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, (2*L) - 1])
        final_x = final_x[:, :, :L, L-1:]
        return final_x

    def relative_logits_1d(self, q, rel_k, H, W, Nh, transpose_mask):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Compute relative logits along one dimension.
        """
        # [B, Nh, H, W, 2*W-1]
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        # Collapse height and heads
        rel_logits = tf.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it back and tile height times
        rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
        # Reshape for adding to the attention logits
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
        return rel_logits

    def relative_logits(self, q, Nh):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Compute relative position logits.
        """
        # [B, Nh, H, W, dk]
        dk = q.shape[-1]
        H = q.shape[2]
        W = q.shape[3]

        # Relative logits in width dimension
        # stddev=dk.value ** -0.5 ?
        key_rel_w = tf.get_variable('key_rel_w', shape=(2*W-1, dk),
                                    initializer=tf.random_normal_initializer(dk.value ** -0.5))

        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, [0, 1, 2, 4, 3, 5])

        # Relative logits in height dimension.
        # For ease, we transpose height and width and repeat the above steps, and transpose to
        # eventually put the logits in the correct position
        key_rel_h = tf.get_variable('key_rel_h', shape=(2*H-1, dk),
                                    initializer=tf.random_normal_initializer(dk.value ** -0.5))

        rel_logits_h = self.relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),
                                               key_rel_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    def split_heads_2d(self, inputs, Nh):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Split channels into multiple heads.
        """
        s = inputs.shape[:-1].as_list()
        s = [x if x is not None else -1 for x in s]
        channels = inputs.shape[-1].value
        ret_shape = s + [Nh, channels // Nh]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0, 3, 1, 2, 4])

    def combine_heads_2d(self, inputs):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Combine heads (inverse of split_heads_2d)
        """
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        a, b = transposed.shape[-2:].as_list()
        ret_shape = transposed.shape[:-2].as_list() + [a*b]
        ret_shape = [x if x is not None else -1 for x in ret_shape]
        return tf.reshape(transposed, ret_shape)

    def compute_flat_qkv(self, inputs, dk, dv, Nh):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Compute flattened queries, keys and values.
        """
        N, H, W, _ = inputs.shape
        qkv = tf.layers.conv2d(inputs, (2*dk) + dv, 1)

        q, k, v = tf.split(qkv, [dk, dk, dv], axis=3)

        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        # Scale query
        dkh = dk // Nh
        dvh = dv // Nh
        q *= (dkh ** -0.5)

        B = N if N.value is not None else -1
        flat_q = tf.reshape(q, [B, Nh, H * W, dkh])
        flat_k = tf.reshape(k, [B, Nh, H * W, dkh])
        flat_v = tf.reshape(v, [B, Nh, H * W, dvh])
        return flat_q, flat_k, flat_v, q

    def augmented_conv2d(self, X, Fout, k=1, dk=256, dv=256, Nh=8, relative=True):
        """
        Code adapted from Attention Augmented Convolutional Networks:
        https://arxiv.org/abs/1904.09925

        Results with:
        dk=256; dv=256; Fout=512 generally best results
        dk=96; Fout=352
        dk=24; Fout=256
        """

        if (dk % Nh != 0) or (dv % Nh != 0):
            raise ValueError("dk or dv is not divisible by Nh")

        if (dk // Nh < 1) or (dv // Nh < 1):
            raise ValueError("(dk or dv) / Nh cannot be less then 1")

        # X has shape [B, H, W, Fin]
        B, H, W, _ = X.shape
        conv_out = tf.layers.conv2d(X, Fout - dv, k)
        # [B, Nh, HW, dvh or dkh]
        flat_q, flat_k, flat_v, q = self.compute_flat_qkv(X, dk, dv, Nh)
        # [B, Nh, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        if relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q, Nh)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = tf.nn.softmax(logits)
        # [B, Nh, HW, dvh]
        attn_out = tf.matmul(weights, flat_v)
        B = B if B.value is not None else -1
        attn_out = tf.reshape(flat_v, [B, Nh, H, W, dv // Nh])
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = tf.layers.conv2d(attn_out, dv, 1)
        return tf.concat([conv_out, attn_out], axis=3)

    def finalize(self, pd, vpred, ph_istate=None): #pylint: disable=W0221
        self.pd = pd
        self.vpred = vpred
        self.ph_istate = ph_istate

    def ensure_observation_is_dict(self, ob):
        if self.ph_ob_keys==[None]:
            return { None: ob }
        else:
            return ob

    def call(self, ob, new, istate):
        """
        Return acs, vpred, neglogprob, nextstate
        """
        raise NotImplementedError

    def initial_state(self, n):
        raise NotImplementedError

    def update_normalization(self, ob):
        pass
