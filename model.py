import tensorflow.keras as k
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam

trajectory_size=28

## Input: agent_state, context_states, map
# Part1: encode agent_state, encode context_states, cnn for map
# part2: concatenate map and context states
# part3: Attention Head with query: agent_state_encoded and key/value: concatenated_context_cnn
# part4: concatenate agent_state_encoded and attention_head_l(i)
# part5: decode the LSTM

def euclidean_distance_loss(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    sq = k.backend.square(k.backend.reshape(y_pred, (-1, 12, 2)) - y_true)
    sq_sum = k.backend.sum(sq, axis=-1)
    return k.backend.sqrt(sq_sum)

# https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
##https://stats.stackexchange.com/questions/319954/whats-the-difference-between-multivariate-gaussian-and-mixture-of-gaussians
## https://stats.stackexchange.com/questions/478625/log-likelihood-of-normal-distribution-why-the-term-fracn2-log2-pi-sigma
def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """
    # k.backend.print_tensor("before y true[0,0,:]:", ytrue[0, 0])
    # k.backend.print_tensor("before y true[0,1,:]::", ytrue[0, 1])
    # k.backend.print_tensor("before y pred[0,0,:]::", ypreds[0, 0])
    # k.backend.print_tensor("before y pred[0,1,:]::", ypreds[0, 1])

    ytrue = k.backend.reshape(ytrue, (-1, 24))

    ypreds = k.backend.reshape(ypreds, (-1, 48))

    # k.backend.print_tensor("after y true[0,0,:]:", ytrue[0, 0:2])
    # k.backend.print_tensor("after y true[0,1,:]::", ytrue[0, 2:4])
    # k.backend.print_tensor("after y pred[0,0,:]::", ypreds[0, 0:2])
    # k.backend.print_tensor("after y pred[0,1,:]::", ypreds[0, 2:4])
    #
    # k.backend.print_tensor("y true[0,0,:]:", ytrue[0, 0])
    # k.backend.print_tensor("y true[0,1,:]::", ytrue[0, 1])
    # k.backend.print_tensor("y pred[0,0,:]::", ypreds[0, 0])
    # k.backend.print_tensor("y pred[0,1,:]::", ypreds[0, 1])

    n_dims = int(ypreds.shape[1] / 2)
    # mu = ypreds[:, 0:n_dims]
    mu = ypreds[:, 0::2] # 24 u --> 12 for x and 12 for y
    # logsigma = ypreds[:, n_dims:]
    logsigma = ypreds[:, 1::2]

    mse = -0.5 * k.backend.sum(k.backend.square((ytrue - mu) / k.backend.exp(logsigma)), axis=1)

    sigma_trace = -k.backend.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return k.backend.mean(-log_likelihood)

def get_attention_head(query, context):
    keys = k.layers.Conv2D(64, (1,1))(context)
    values = k.layers.Conv2D(64, (1,1))(context)
    ## need to put scale = sqrt(d)
    keys = k.layers.Reshape((32*32, 64))(keys)
    # keys = k.layers.Permute((2, 1))(keys)
    values = k.layers.Reshape((32*32, 64))(values)
    query = k.layers.Reshape((1, 64))(query)
    combined = k.layers.Attention()([query, values, keys])

    # mult_scores_values = tf.reshape(mult_scores_values, (-1, 1, 64))
    # print("weights", mult_scores_values.shape)
    # mult = tf.multiply(weights,values)
    # print("mult", mult.shape)
    # combined = tf.reduce_sum(mult_scores_values, axis=1)
    combined = k.backend.squeeze(combined, axis=1)
    return combined


def get_map_cnn_model(map_BGR):
    x = k.layers.Conv2D(32, (3, 3), strides=(2, 2))(map_BGR)
    x = k.layers.ZeroPadding2D((1,1))(x)
    x = k.layers.Conv2D(32, (3, 3), strides=(2, 2))(x)
    x = k.layers.ZeroPadding2D((1,1))(x)
    x = k.layers.Conv2D(64, (3, 3), strides=(2, 2))(x)
    x = k.layers.ZeroPadding2D((1,1))(x)
    x = k.layers.Conv2D(64, (3, 3), strides=(2, 2))(x)
    return x


def build_model_mha_jam():
    ## input
    # k.backend.set_floatx('float16')
    L=1
    agent_state_inp = k.Input(shape=(trajectory_size, 5))
    agent_context_inp = k.Input(shape=(32, 32, trajectory_size, 5))
    # agent_map_inp = k.Input(shape=(500,500,3))
    agent_map_inp = k.Input(shape=(1024, 1024, 3), dtype="float16")
    ## input embedding
    embedding_layer = k.layers.Dense(64, activation='relu')

    agent_state = k.layers.TimeDistributed(embedding_layer)(agent_state_inp)
    agent_context = k.layers.TimeDistributed(
            k.layers.TimeDistributed(
                    k.layers.TimeDistributed(
                            embedding_layer)))(agent_context_inp)

    ## input encoding
    trajectory_encoder = k.layers.LSTM(64)

    agent_state_encoded = trajectory_encoder(agent_state)
    agent_context_encoded = k.layers.TimeDistributed(
            k.layers.TimeDistributed(
                    trajectory_encoder))(agent_context)

    ## add here map details
    map_features_extractor = k.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(1024, 1024, 3)
    )

    for layer in map_features_extractor.layers:
        layer.trainable = False

    agent_map = map_features_extractor(agent_map_inp)
    # agent_map = get_map_cnn_model(agent_map_inp)

    agent_context_encoded = k.backend.concatenate([agent_map, agent_context_encoded])
    ## L Attention Heads

    outs = []
    # zls = []
    trajectory_decoder_0 = k.layers.LSTM(128, return_sequences=True)
    trajectory_decoder_1 = k.layers.Dense(2) # set to 4 for gaussian!
    # trajectory_decoder_0 = k.layers.Dense(128, activation='relu')
    # trajectory_decoder_1 = k.layers.Dense(24)

    for i in range(L):
        attention_i_result = get_attention_head(agent_state_encoded, agent_context_encoded)
        # can be axis 0 if 0 is not for batch .. need checking
        z_l = k.layers.Concatenate(axis=1)([agent_state_encoded, attention_i_result])
        z_l = k.layers.RepeatVector(12)(z_l)

        # zls.append(z_l)
        # for simplicity, I use Dense Layer, instead, the paper uses an LSTM
        out_l = trajectory_decoder_0 (z_l)
        out_l = trajectory_decoder_1(out_l)
        # out_l = k.layers.Reshape((48,))(out_l)
        outs.append(out_l)

    # x = k.layers.Dense(200)(tf.concat(zls, axis=0))
    # out_prob = k.layers.Dense(L, activation='softmax')(x)
    # need to review the out location for this
    # outs.append(out_prob)

    model = k.Model(inputs=[agent_state_inp, agent_context_inp, agent_map_inp], outputs=outs)
    model.summary()
    # model.compile("adam", loss=gaussian_nll)
    model.compile("adam", loss=euclidean_distance_loss)

    # k.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=400
    # )
    return model


def build_model_mha_sam():
    raise NotImplemented("do not use this for now .. untested");
    ## input
    # k.backend.set_floatx('float16')
    L=1
    agent_state_inp = k.Input(shape=(trajectory_size, 5))
    agent_context_inp = k.Input(shape=(32, 32, trajectory_size, 5))
    # agent_map_inp = k.Input(shape=(500,500,3))
    agent_map_inp = k.Input(shape=(1024, 1024, 3), dtype="float16")
    ## input embedding
    embedding_layer = k.layers.Dense(64, activation='relu')

    agent_state = k.layers.TimeDistributed(embedding_layer)(agent_state_inp)
    agent_context = k.layers.TimeDistributed(
            k.layers.TimeDistributed(
                    k.layers.TimeDistributed(
                            embedding_layer)))(agent_context_inp)

    ## input encoding
    trajectory_encoder = k.layers.LSTM(64)

    agent_state_encoded = trajectory_encoder(agent_state)
    agent_context_encoded = k.layers.TimeDistributed(
            k.layers.TimeDistributed(
                    trajectory_encoder))(agent_context)

    ## add here map details
    map_features_extractor = k.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(1024, 1024, 3)
    )

    for layer in map_features_extractor.layers:
        layer.trainable = False

    agent_map = map_features_extractor(agent_map_inp)
    # agent_map = get_map_cnn_model(agent_map_inp)

    # agent_context_encoded = k.backend.concatenate([agent_map, agent_context_encoded])
    ## L Attention Heads

    outs = []
    # zls = []
    # trajectory_decoder_0 = k.layers.LSTM(1trajectory_size)
    # trajectory_decoder_1 = k.layers.Dense(2, activation='relu')
    trajectory_decoder_0 = k.layers.Dense(64, activation='relu')
    trajectory_decoder_1 = k.layers.Dense(24, activation='relu')

    for i in range(L):
        attention_i_result_static = get_attention_head(agent_map, agent_context_encoded)
        attention_i_result_dynamic = get_attention_head(agent_state_encoded, agent_context_encoded)
        attention_i_result = k.backend.concatenate([attention_i_result_static, attention_i_result_dynamic])

        # can be axis 0 if 0 is not for batch .. need checking
        z_l = k.layers.Concatenate(axis=1)([agent_state_encoded, attention_i_result])
        # zls.append(z_l)
        # for simplicity, I use Dense Layer, instead, the paper uses an LSTM
        out_l = trajectory_decoder_0 (z_l)
        out_l = trajectory_decoder_1(out_l)
        outs.append(out_l)

    # x = k.layers.Dense(200)(tf.concat(zls, axis=0))
    # out_prob = k.layers.Dense(L, activation='softmax')(x)
    # need to review the out location for this
    # outs.append(out_prob)

    model = k.Model(inputs=[agent_state_inp, agent_context_inp, agent_map_inp], outputs=outs)
    model.summary()
    model.compile("adam", loss=euclidean_distance_loss)

    # k.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=400
    # )
    return model
