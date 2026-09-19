# Third-party notices

## Wuji retargeting

The kinematic retargeting subset under `src/wuji_lfh/_vendor/wuji_retargeting/`
comes from [wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting),
revision `531f6ed4250b475d2e9231f54e988fc9b1c5b4ea` (version `2026.8.17`).
It includes the analytical adaptive and vector optimizers, robot wrapper,
coordinate transforms, and retargeter. The optimizer mathematics are unchanged;
application imports use the local namespace. The MIT license is included in
that directory. Examples, hardware SDK integration, visualization, and demo
recordings are not bundled.

## Wuji kinematic models

The left/right hand URDFs come from
[wuji-technology/wuji-description](https://github.com/wuji-technology/wuji-description),
revision `8271644a78d69ed9a4adcf9165d882c64ad33dfa`.
Their MIT license is included under the bundled `wuji-description/` directory.
Visual/collision geometry is omitted: retargeting uses only joint/link
kinematics through Pinocchio; rendering uses wuji-mjlab assets.

## Data and MANO

Human demonstrations and licensed MANO model files are user-provided, ignored
by Git, and are not part of this repository's software license.
