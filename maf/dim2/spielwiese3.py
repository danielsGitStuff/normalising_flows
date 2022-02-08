if __name__ == '__main__':
    typ = 'int'
    # s = "take_t_sig  take_t_noi  take_v_sig  take_v_noi"
    s = 'size_nf_t_noi  size_nf_t_sig  size_nf_v_noi  size_nf_v_sig'
    # s = 'take_t_sig  take_t_noi  take_v_sig  take_v_noi'
    ss = [s for s in s.split('  ')]
    print(ss)

    print('')
    print('')

    for s in ss:
        print(f"{s}: {typ} = NotProvided(),")

    print('')
    print('')

    for s in ss:
        print(f"self.{s}: {typ} = {s}")

    print('')
    print('')

    for s in ss:
        print(f"{s}={s},")
