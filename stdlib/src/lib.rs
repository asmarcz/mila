use std::{
    ffi::{c_char, CStr, c_longlong},
    io::{stdout, Write, stdin},
};

#[no_mangle]
pub extern "C" fn mila_inc(int: *mut c_longlong) {
    unsafe {
        *int = *int - 1;
    }
}

#[no_mangle]
pub extern "C" fn mila_dec(int: *mut c_longlong) {
    unsafe {
        *int = *int - 1;
    }
}

#[no_mangle]
pub extern "C" fn mila_int(double: f64) -> i64 {
    double as i64
}

#[no_mangle]
pub extern "C" fn mila_float(int: i64) -> f64 {
    int as f64
}

#[no_mangle]
pub extern "C" fn mila_write(int: i64) {
    print!("{}", int);
    let _ = stdout().flush();
}

#[no_mangle]
pub extern "C" fn mila_writeln(int: i64) {
    println!("{}", int);
}

#[no_mangle]
pub extern "C" fn mila_dbl_write(double: f64) {
    print!("{}", double);
    let _ = stdout().flush();
}

#[no_mangle]
pub extern "C" fn mila_dbl_writeln(double: f64) {
    println!("{}", double);
}

#[no_mangle]
pub extern "C" fn mila_str_write(ptr: *const c_char) {
    unsafe {
        print!("{}", CStr::from_ptr(ptr).to_str().unwrap());
        let _ = stdout().flush();
    }
}

#[no_mangle]
pub extern "C" fn mila_str_writeln(ptr: *const c_char) {
    unsafe {
        println!("{}", CStr::from_ptr(ptr).to_str().unwrap());
    }
}

#[no_mangle]
pub extern "C" fn mila_readln(int: *mut c_longlong) {
    let mut buf = String::new();
    stdin().read_line(&mut buf).unwrap();
    let read: i64 = buf.trim().parse().unwrap();
    unsafe {
        *int = read;
    }
}
