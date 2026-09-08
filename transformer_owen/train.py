from decoder_transformer import build_transformer




if __name__ == '__main__':
    net = build_transformer(4, 350)
    print(net)