import tensorflow.keras as k


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


def get_attention_head(query, context):
    keys = k.layers.Conv2D(64, (1,1))(context)
    values = k.layers.Conv2D(64, (1,1))(context)
    ## need to put scale = sqrt(d)
    keys = k.layers.Reshape((50*50, 64))(keys)
    # keys = k.layers.Permute((2, 1))(keys)
    values = k.layers.Reshape((50*50, 64))(values)
    query = k.layers.Reshape((1, 64))(query)
    combined = k.layers.Attention()([query, values, keys])

    # mult_scores_values = tf.reshape(mult_scores_values, (-1, 1, 64))
    # print("weights", mult_scores_values.shape)
    # mult = tf.multiply(weights,values)
    # print("mult", mult.shape)
    # combined = tf.reduce_sum(mult_scores_values, axis=1)
    combined = k.backend.squeeze(combined, axis=1)
    return combined



def build_model():
    ## input
    # k.backend.set_floatx('float16')
    L=1
    agent_state_inp = k.Input(shape=(8, 5))
    agent_context_inp = k.Input(shape=(50, 50, 8, 5), dtype="float16")
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
    pass

    ## L Attention Heads

    outs = []
    # zls = []
    # trajectory_decoder = k.layers.LSTM(128)
    trajectory_decoder_0 = k.layers.Dense(64, activation='relu')
    trajectory_decoder_1 = k.layers.Dense(24, activation='relu')

    for i in range(L):
        attention_i_result = get_attention_head(agent_state_encoded, agent_context_encoded)
        # can be axis 0 if 0 is not for batch .. need checking
        z_l = k.layers.Concatenate(axis=1)([agent_state_encoded, attention_i_result])
        # zls.append(z_l)
        # for simplicity, I use Dense Layer, instead, the paper uses an LSTM
        out_l = trajectory_decoder_0(z_l)
        out_l = trajectory_decoder_1(out_l)
        outs.append(out_l)

    # x = k.layers.Dense(200)(tf.concat(zls, axis=0))
    # out_prob = k.layers.Dense(L, activation='softmax')(x)
    # need to review the out location for this
    # outs.append(out_prob)

    model = k.Model(inputs=[agent_state_inp, agent_context_inp], outputs=outs)
    model.summary()
    model.compile("adam", loss=euclidean_distance_loss)
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=400
    # )
    return model
