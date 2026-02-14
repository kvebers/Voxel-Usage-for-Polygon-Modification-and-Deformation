def interpolate_camera(cam_keys, t):
    if not cam_keys:
        return [20, 6, 0], [0, 0, 0], 45.0
    if t <= cam_keys[0]['time']:
        c = cam_keys[0]
        return c['position'], c['target'], c['fov']
    if t >= cam_keys[-1]['time']:
        c = cam_keys[-1]
        return c['position'], c['target'], c['fov']

    for i in range(len(cam_keys) - 1):
        c0, c1 = cam_keys[i], cam_keys[i + 1]
        if c0['time'] <= t <= c1['time']:
            f = (t - c0['time']) / max(c1['time'] - c0['time'], 1e-6)
            pos = [c0['position'][j] + f * (c1['position'][j] - c0['position'][j]) for j in range(3)]
            tgt = [c0['target'][j]   + f * (c1['target'][j]   - c0['target'][j])   for j in range(3)]
            fov = c0['fov'] + f * (c1['fov'] - c0['fov'])
            return pos, tgt, fov

    c = cam_keys[-1]
    return c['position'], c['target'], c['fov']