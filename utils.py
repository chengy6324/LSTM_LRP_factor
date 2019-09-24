def sub_y(d):
    if isinstance(d, str):
        return d[:4]
    else:
        return d.strftime('%Y')


def sub_m(d):
    if isinstance(d, str):
        return d[:7]
    else:
        return d.strftime('%Y-%m')


def sub_d(d):
    if isinstance(d, str):
        return d[:10]
    else:
        return d.strftime('%Y-%m-%d')


def cal_index(r, u):
    index_start = [r.index(i, 0) for i in u]
    index_last = [i for i, v in enumerate(r) if v == u[-1]][-1]
    index_end = [i - 1 for i in index_start]
    index_end.pop(0)
    index_end.append(index_last)
    return index_start, index_end


def to_percent(temp, position):
    return '%1.0f' % (100*temp) + '%'


def to_percent_1(temp, position):
    return '%1.1f' % (100*temp) + '%'


def to_percent_2(temp, position):
    return '%1.2f' % (100*temp) + '%'