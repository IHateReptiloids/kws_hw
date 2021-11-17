import torch
import torch.nn.functional as F

from src.configs import DefaultConfig
from .crnn import CRNN


class StreamingCRNN(CRNN):
    def __init__(self, window_length, config: DefaultConfig):
        '''
        window_length defines number of frames saved before
        attention layer
        '''
        super().__init__(config)

        self._n_frame_stride = config.stride[1]
        self._n_receptive_frames = config.kernel_size[1]
        self._frames_processed = 0
        self._context_size = 0

        self.register_buffer('last_spec_frames',
                             torch.zeros(config.n_mels,
                                         self._n_receptive_frames))

        n_hidden_states = (int(config.bidirectional) + 1) * \
            config.gru_num_layers
        self.register_buffer('gru_hidden_states',
                             torch.zeros(n_hidden_states, config.hidden_size))

        context_dim = (int(config.bidirectional) + 1) * config.hidden_size
        self.register_buffer('context',
                             torch.zeros(window_length, context_dim))

    def load_crnn_state_dict(self, state_dict):
        '''
        state_dict is a CRNN state_dict, not StreamingCRNN state_dict!
        '''
        old_state_dict = self.state_dict()
        old_state_dict.update(state_dict)
        self.load_state_dict(old_state_dict)

    @torch.no_grad()
    def process_frame(self, x):
        '''
        x has a shape of (n_features,)
        '''
        self._update_last_spec_frames(x)
        if self._frames_processed >= self._n_receptive_frames and \
                (self._frames_processed - self._n_receptive_frames) % \
                self._n_frame_stride == 0:
            gru_output = self._update_gru_states()
            self._update_context(gru_output)

        return self._make_prediction()

    def reset_streaming(self):
        self.last_spec_frames.fill_(0)
        self.gru_hidden_states.fill_(0)
        self.context.fill_(0)
        self._frames_processed = 0
        self._context_size = 0

    def _make_prediction(self):
        context = self.context[-self._context_size:]
        logits = self.classifier(self.attention(context))
        return F.softmax(logits, dim=-1)

    def _update_context(self, x):
        context = self.context.clone()
        self.context[:-1, :] = context[1:, :]
        self.context[-1, :] = x
        self._context_size += 1

    def _update_gru_states(self):
        x = self.conv(self.last_spec_frames[None, None, :, :])
        x = x.squeeze()
        assert x.shape == (self.gru.input_size,)

        gru_output, hid_states = self.gru(x[None, None, :],
                                          self.gru_hidden_states.unsqueeze(1))
        self.gru_hidden_states = hid_states.squeeze()
        return gru_output.squeeze()

    def _update_last_spec_frames(self, x):
        last_spec_frames = self.last_spec_frames.clone()
        self.last_spec_frames[:, :-1] = last_spec_frames[:, 1:]
        self.last_spec_frames[:, -1] = x
        self._frames_processed += 1
