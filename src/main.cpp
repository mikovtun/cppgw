// cppgw entry point (Story 2: input framework).
//
// Run modes:
//   cppgw -i input_file -d data.h5     normal calculation (order of -i / -d free)
//   cppgw input_file data.h5           same, positional form
//   cppgw -t  |  cppgw -test           run the built-in test suite (exclusive of -i/-d)
//   cppgw -h                           print the help page
//
// An invalid invocation (missing args, unknown flags, -t combined with -i/-d or
// positional arguments) prints the help page and exits with code 2.

#include "main.hpp"
#include "tests.hpp"

#include <highfive/highfive.hpp>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_help(const char* prog) {
  std::cout <<
    "cppgw - contracted Green's-function code\n"
    "\n"
    "Usage:\n"
    "  " << prog << " -i INPUT -d DATA     run a calculation with the given input and data files\n"
    "  " << prog << " INPUT DATA           equivalent positional form\n"
    "  " << prog << " -t | -test           run the built-in test suite (no other arguments allowed)\n"
    "  " << prog << " -h                   show this help and exit\n"
    "\n"
    "Options:\n"
    "  -i INPUT    input file specifying the calculation parameters (parsed as text)\n"
    "  -d DATA     HDF5 file containing the numerical data to read in\n"
    "  -t, -test   run the built-in test suite; exclusive with -i, -d, and positional arguments\n"
    "  -h          show this help page\n"
    "\n"
    "Examples:\n"
    "  " << prog << " -i test.in -d rhf_df.h5\n"
    "  " << prog << " -d rhf_df.h5 -i test.in\n"
    "  " << prog << " test.in rhf_df.h5\n"
    "  " << prog << " -t\n";
}

struct Options {
  bool        help  = false;
  bool        test  = false;
  std::string input;
  std::string data;
};

// Returns true if argv is a valid invocation; false otherwise.
bool parse_args(int argc, char** argv, Options& opt) {
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h") {
      opt.help = true;
    }
    else if (a == "-t" || a == "-test") {
      if (opt.test)
        return false;                          // redundant flag
      opt.test = true;
    }
    else if (a == "-i") {
      if (i + 1 >= argc)
        return false;                          // -i needs a value
      if (!opt.input.empty())
        return false;                          // input given twice
      opt.input = argv[++i];
    }
    else if (a == "-d") {
      if (i + 1 >= argc)
        return false;                          // -d needs a value
      if (!opt.data.empty())
        return false;                          // data given twice
      opt.data = argv[++i];
    }
    else if (!a.empty() && a.front() == '-') {
      return false;                            // unknown option (e.g. --baz)
    }
    else if (!a.empty()) {
      pos.push_back(a);
    }
    else {
      return false;                            // empty argument
    }
  }

  if (opt.help)
    return true;                               // -h takes precedence, exit 0

  if (opt.test) {
    // Test mode is exclusive with any calculation input.
    if (!opt.input.empty() || !opt.data.empty() || !pos.empty())
      return false;
    return true;
  }

  // Normal calculation mode: exactly (INPUT, DATA), given either as two
  // positional arguments or as the -i / -d flag pair.
  if (pos.size() == 2 && opt.input.empty() && opt.data.empty()) {
    opt.input = pos[0];
    opt.data  = pos[1];
    return true;
  }
  if (pos.empty() && !opt.input.empty() && !opt.data.empty())
    return true;
  return false;                                // incomplete / ambiguous / extra args
}

} // namespace

namespace cppgw {

// Check (on entry) that the input file is usable: it must exist and be
// readable as text. The input format is extension-independent.
//
// Parsing of the input file's contents is intentionally not implemented yet
// (see story 03); this function is a stub that will grow into the real parser.
void check_input_file(const std::string& path) {
  std::ifstream in(path);                       // default open mode: text, read
  if (!in.is_open())
    throw std::logic_error("cppgw: cannot open input file '" + path + "'");
  // TODO(story 03): parse the input file contents (assumed text encoding).
}

// Check that the data file is a healthy HDF5 file by opening it read-only
// with HighFive. HighFive throws (FileInvalid / SystemError /
// FileDriverException) for a missing, corrupt, or non-HDF5 file.
void check_data_file(const std::string& path) {
  HighFive::File f(path, HighFive::File::ReadOnly);   // may throw
  (void)f;
}

} // namespace cppgw

int main(int argc, char** argv) {
  const char* prog = (argc > 0 && argv[0] != nullptr) ? argv[0] : "cppgw";

  Options opt;
  if (!parse_args(argc, argv, opt)) {
    print_help(prog);                            // invalid invocation -> help page
    return 2;
  }
  if (opt.help) {
    print_help(prog);
    return 0;
  }
  if (opt.test) {
    return cppgw::run_tests();                   // throws on failure
  }

  // Normal calculation mode: validate the inputs, then (later stories) run
  // the calculation described by the input file.
  cppgw::check_input_file(opt.input);
  cppgw::check_data_file(opt.data);
  // TODO: entry point for the (to-be-determined) calculation goes here.
  return 0;
}
