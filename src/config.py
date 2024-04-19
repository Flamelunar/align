from configparser import ConfigParser
import sys
import os


# sys.path.append('..')


class Configurable(object):
    def __init__(self, config_file, extra_args):
        self.config_file = config_file
        print("read config from " + config_file)
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._conf = config
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        assert self.model_dir.endswith('/')
        config.write(open(self.model_dir + self.config_file + '.bak', 'w'))
        print('Loaded config file successfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    # Run control
    # @property
    # def num_buckets_train(self):
    #     return self._config.getint('Run', 'num_buckets_train')
    # @property
    # def num_buckets_valid(self):
    #     return self._config.getint('Run','num_buckets_valid')
    # @property
    # def num_buckets_test(self):
    #     return self._config.getint('Run','num_buckets_test')

    @property
    def sent_num_one_batch(self):
        return self._conf.getint('Run', 'sent_num_one_batch')

    @property
    def word_num_one_batch(self):
        return self._conf.getint('Run', 'word_num_one_batch')

    @property
    def max_bucket_num(self):
        return self._conf.getint('Run', 'max_bucket_num')  # negative means not using bucket

    @property
    def is_train(self):
        return self._conf.getint('Run', 'is_train') > 0

    @property
    def is_test(self):
        return self._conf.getint('Run', 'is_test') > 0

    @property
    def device(self):
        return self._conf.get('Run', 'device')

    @property
    def dict_dir(self):
        return self._conf.get('Run', 'dict_dir')

    @property
    def word_freq_cutoff(self):
        return self._conf.getint('Run', 'word_freq_cutoff')

    @property
    def model_dir(self):
        return self._conf.get('Run', 'model_dir')

    @property
    def ext_word_emb_full_path(self):
        return self._conf.get('Run', 'ext_word_emb_full_path')

    @property
    def ext_word_dict_full_path(self):
        return self._conf.get('Run', 'ext_word_dict_full_path')

    @property
    def inst_num_max(self):
        return self._conf.getint('Run', 'inst_num_max')

    @property
    def is_shared_lstm(self):
        return self._conf.getint('Run', 'is_shared_lstm') > 0

    @property
    def is_gate_lstm(self):
        return self._conf.getint('Run', 'is_gate_lstm') > 0

    @property
    def is_diff_loss(self):
        return self._conf.getint('Run', 'is_diff_loss') > 0

    @property
    def is_domain_emb(self):
        return self._conf.getint('Run', 'is_domain_emb') > 0

    @property
    def is_adversary(self):
        return self._conf.getint('Run', 'is_adversary') > 0

    @property
    def is_charlstm(self):
        return self._conf.getint('Run', 'is_charlstm') > 0

    @property
    def is_multi(self):
        return self._conf.getint('Run', 'is_multi') > 0

    @property
    def model_eval_num(self):
        return self._conf.getint('Test', 'model_eval_num')

    @property
    def test_files(self):
        return self._conf.get('Train', 'test_files')

    @property
    def train_files(self):
        return self._conf.get('Train', 'train_files')  # use ; to split multiple training datasets

    @property
    def dev_files(self):
        return self._conf.get('Train', 'dev_files')

    @property
    def unlabel_train_files(self):
        return self._conf.get('Train', 'unlabel_train_files')  # unlabel_train_files

    @property
    def is_dictionary_exist(self):
        return self._conf.getint('Train', 'is_dictionary_exist') > 0

    @property
    def train_max_eval_num(self):
        return self._conf.getint('Train', 'train_max_eval_num')

    @property
    def save_model_after_eval_num(self):
        return self._conf.getint('Train', 'save_model_after_eval_num')

    @property
    def train_stop_after_eval_num_no_improve(self):
        return self._conf.getint('Train', 'train_stop_after_eval_num_no_improve')

    @property
    def eval_every_update_step_num(self):
        return self._conf.getint('Train', 'eval_every_update_step_num')

    @property
    def save_model_after_eval_num(self):
        return self._conf.getint('Train', 'save_model_after_eval_num')

    @property
    def lstm_layer_num(self):
        return self._conf.getint('Network', 'lstm_layer_num')

    @property
    def word_emb_dim(self):
        return self._conf.getint('Network', 'word_emb_dim')

    @property
    def tag_emb_dim(self):
        return self._conf.getint('Network', 'tag_emb_dim')

    @property
    def domain_emb_dim(self):
        return self._conf.getint('Network', 'domain_emb_dim')

    @property
    def domain_size(self):
        return self._conf.getint('Network', 'domain_size')

    @property
    def emb_dropout_ratio(self):
        return self._conf.getfloat('Network', 'emb_dropout_ratio')

    @property
    def lstm_hidden_dim(self):
        return self._conf.getint('Network', 'lstm_hidden_dim')

    @property
    def lstm_input_dropout_ratio(self):
        return self._conf.getfloat('Network', 'lstm_input_dropout_ratio')

    @property
    def lstm_hidden_dropout_ratio_for_next_timestamp(self):
        return self._conf.getfloat('Network', 'lstm_hidden_dropout_ratio_for_next_timestamp')

    @property
    def mlp_output_dim_arc(self):
        return self._conf.getint('Network', 'mlp_output_dim_arc')

    @property
    def mlp_output_dim_rel(self):
        return self._conf.getint('Network', 'mlp_output_dim_rel')

    @property
    def mlp_input_dropout_ratio(self):
        return self._conf.getfloat('Network', 'mlp_input_dropout_ratio')

    @property
    def mlp_output_dropout_ratio(self):
        return self._conf.getfloat('Network', 'mlp_output_dropout_ratio')

    @property
    def learning_rate(self):
        return self._conf.getfloat('Optimizer', 'learning_rate')

    @property
    def decay(self):
        return self._conf.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._conf.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._conf.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._conf.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._conf.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._conf.getfloat('Optimizer', 'clip')

    @property
    def adversary_lambda_loss(self):
        return self._conf.getfloat('Optimizer', 'adversary_lambda_loss')

    @property
    def diff_bate_loss(self):
        return self._conf.getfloat('Optimizer', 'diff_bate_loss')
