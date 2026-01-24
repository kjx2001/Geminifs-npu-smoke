#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <libunwind.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <exception>

void print_stacktrace()
{
    unw_cursor_t cursor;
    unw_context_t context;
    unw_word_t ip, sp, offset;
    char funcname[256];

    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    fprintf(stderr, "=== Stack trace ===\n");
    int frame = 0;
    while (unw_step(&cursor) > 0)
    {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_reg(&cursor, UNW_REG_SP, &sp);
        if (unw_get_proc_name(&cursor, funcname, sizeof(funcname), &offset) == 0)
        {
            int status;
            char *demangled = abi::__cxa_demangle(funcname, NULL, 0, &status);
            const char *final_funcname = (status == 0 && demangled) ? demangled : funcname;
            fprintf(stderr, "#%d 0x%lx: %s+0x%lx\n", frame, ip, final_funcname, offset);
            if (demangled)
                free(demangled);
        }
        else
        {
            fprintf(stderr, "#%d 0x%lx: --unknown--\n", frame, ip);
        }
        frame++;
    }
    fprintf(stderr, "===================\n");
}

static void signal_handler(int sig)
{
    printf("Caught signal %d\n", sig);
    print_stacktrace();
    _exit(1); // exit immediately, do not invoke default abort handler
}

static void setup_signal_handler()
{
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND | SA_NODEFER;
    int fatal_signals[] = {
        SIGSEGV, // 段错误
        SIGILL,  // 非法指令
        SIGFPE,  // 浮点异常
        SIGABRT, // abort
        SIGBUS,  // 总线错误
        SIGTRAP, // 调试断点
        SIGSYS,  // 非法系统调用
        SIGQUIT, // Ctrl+
        SIGXFSZ  // 文件超长
    };
    int nsigs = sizeof(fatal_signals) / sizeof(fatal_signals[0]);
    for (int i = 0; i < nsigs; ++i)
    {
        sigaction(fatal_signals[i], &sa, NULL);
    }
}

static void terminate_handler()
{
    fprintf(stderr, "Program terminated due to uncaught exception!\n");
    print_stacktrace();
    std::_Exit(1);
}

static void setup_terminate_handler()
{
    std::set_terminate(terminate_handler);
}

void setup_backtrace()
{
    setup_signal_handler();
    setup_terminate_handler();
}