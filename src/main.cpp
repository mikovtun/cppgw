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
#include "input.hpp"
#include "tests.hpp"

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
  bool        missing_input = false;
  bool        missing_data  = false;
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
  if (pos.empty()) {
    if (opt.data.empty() && !opt.input.empty()) {
      opt.missing_data = true;     // input given, no data -> one-line error
      return true;
    }
    if (opt.input.empty() && !opt.data.empty()) {
      opt.missing_input = true;    // data given, no input -> one-line error
      return true;
    }
  }
  if (pos.size() == 1 && opt.input.empty() && opt.data.empty()) {
    opt.input = pos[0];
    opt.missing_data = true;       // one positional = input only
    return true;
  }
  return false;                                // no args, or malformed mix -> help page
}

} // namespace

namespace cppgw {

// Run the calculation selected by the input file. The InputCatalog owns
// parsing, defaults, sanitization, and dataset access; per-calc branches consume
// typed requirement objects.
int run_calculation(const std::string& input_path, const std::string& data_path) {
  InputCatalog catalog = InputCatalog::from_file(input_path, data_path);

  switch (catalog.resolved().calc) {
    case Calc::Gf2: {
      Gf2Input gf2 = catalog.require_gf2();
      Gf2InitialGuess guess = make_gf2_initial_guess(gf2);

      const size_t n_ao = gf2.hcore.dims()[0].dim;
      const size_t n_ri = gf2.eri3.dims()[2].dim;
      std::cout << "cppgw: calculation " << calc_name(catalog.resolved().calc) << "\n";
      std::cout << "  data file: " << catalog.hdf5_path() << "\n";
      std::cout << "  AO dimension: " << n_ao << ", RI dimension: " << n_ri << "\n";
      std::cout << "  beta = " << gf2.beta
                << ", matsubara_half_n = " << gf2.matsubara_half_n
                << ", total Matsubara points = " << guess.green.size() << "\n";
      std::cout << "  mu = " << gf2.mu << "\n";
      std::cout << "  Tr(P S) electron-count diagnostic = " << guess.electron_count << "\n";
      std::cout << "  eta = " << gf2.eta
                << (gf2.eta_was_supplied ? " (user-supplied, unused here)" : " (default 1e-5, unused here)") << "\n";
      std::cout << "  constructed AO Matsubara initial Green's function (non-self-consistent; no dynamic self-energy, Fourier transform, or chemical-potential search)\n";

      const std::string out_path = "cppgw_output.h5";
      write_gf2_output_hdf5(gf2, guess, out_path);
      std::cout << "  wrote g0, sigma_hf, and input parameters to " << out_path << "\n";
      return 0;
    }
  }
  std::cerr << "cppgw: no implementation for calculation '" 
              << calc_name(catalog.resolved().calc) << "'\n";
  return 2;
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
  if (opt.missing_input) {
    std::cerr << "cppgw: an input file is required (none given; see -h for usage)\n";
    return 2;
  }
  if (opt.missing_data) {
    std::cerr << "cppgw: a data file is required (none given; see -h for usage)\n";
    return 2;
  }
  if (opt.test) {
    return cppgw::run_tests();                   // throws on failure
  }

  // Normal calculation mode (Story 05): the InputCatalog owns parsing,
  // defaults, sanitization, and dataset access; the calc branch consumes the
  // typed requirement object. Calculation/data failures print an actionable
  // message and exit 2, and do NOT print the help page (help is reserved for
  // invalid invocations above).
  try {
    return cppgw::run_calculation(opt.input, opt.data);
  } catch (const std::exception& e) {
    std::cerr << "cppgw: " << e.what() << '\n';
    return 2;
  }
}
