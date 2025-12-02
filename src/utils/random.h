// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 tomocam contributors

#ifndef UTILS_RANDOM__H
#define UTILS_RANDOM__H

#include <random>

namespace tomocam::utils {
    class NPRandom {
      private:
        std::mt19937 gen_;

      public:
        NPRandom() { gen_ = std::mt19937(5489u); }

        template <typename T>
        T rand() {
            int a = gen_() >> 5;
            int b = gen_() >> 6;
            return static_cast<T>((a * 67108864.0 + b) / 9007199254740992.0);
        }
    };
} // namespace tomocam::utils
#endif // UTILS_RANDOM__H
