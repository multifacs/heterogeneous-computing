#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace sycl;

int main() {
    vector<platform> platforms = platform::get_platforms();
    for (size_t i = 0; i < platforms.size(); i++) {
        cout << "Platform number " << i << ": " << platforms[i].get_info<info::platform::name>() << "\n";
        vector<device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++) {
            cout << "\tDevice number " << j << ": " << devices[j].get_info<info::device::name>() << "\n";
        }
    }
    cout << "\n";

    constexpr int globalSize = 3;
    for (size_t i = 0; i < platforms.size(); i++) {
        vector<device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++) {
            cout << devices[j].get_info<info::device::name>() << "\n";
            try {
                queue q(devices[j]);
                q.submit([&](handler &h) {
                    stream out(1024, 80, h);
                    h.parallel_for(range<1>(globalSize), [=](id<1> item) {
                        out << " -- Item: " << item.get(0) << " Hello from platform " << i << " and device " << j << "\n";
                    });
                });
                q.wait();
            } catch (std::exception e) {
                cout << e.what() << "\n";
            }
        }
        cout << "\n";
    }
}