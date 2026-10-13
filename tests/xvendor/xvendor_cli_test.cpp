#include "xvendor_cli.hpp"
#include <cstdio>
#include <initializer_list>
#include <vector>

static xvendor::Arguments parse(std::initializer_list<const char*> values) {
    std::vector<char*> args;
    args.push_back(const_cast<char*>("probe"));
    for (auto value : values) args.push_back(const_cast<char*>(value));
    return xvendor::arguments(static_cast<int>(args.size()), args.data());
}
static void check(bool ok) { if (!ok) throw std::runtime_error("CLI check failed"); }
int main() {
    try {
        check(parse({"--region-bytes"}).bytes() == xvendor::Layout::standard().bytes);
        check(parse({"--region-bytes", "--mode", "large-registration"}).bytes() == size_t{32} * 1024 * 1024 * 1024);
        auto args = parse({"--role", "worker", "--fd", "0", "--rounds", "1", "--mode", "spin"});
        check(args.fd == 0 && args.rounds == 1 && args.role == "worker");
        for (auto values : {std::initializer_list<const char*>{"--fd", "-1"}, {"--region-bytes", "--rounds", "0"},
             {"--region-bytes", "--rounds", "16777216"}, {"--region-bytes", "--mode", "unknown"},
             {"--role", "bogus", "--fd", "1"}, {"--fd"}, {"--region-bytes", "--fd", "0junk"},
             {"--region-bytes", "--arena-mib", "9999999999999999999999999"}}) {
            bool rejected = false;
            try { parse(values); } catch (const std::exception&) { rejected = true; }
            check(rejected);
        }
        std::puts("PASS: bounded probe CLI and pre-GPU region-size query");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
